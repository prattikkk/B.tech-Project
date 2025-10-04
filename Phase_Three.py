#!/usr/bin/env python3
"""Phase 3 - Real-time IoT Botnet Anomaly Detection via MQTT (production-minded)

Single-source implementation. This file can run directly (python Phase_Three.py)
or be accessed via the package shim `pipeline.phases.phase_three:main` which simply
executes this file and calls its `main()` function. The shim is created so earlier
code referencing `from pipeline.phases.phase_three import main` still works after
introducing a package layout.
"""

# Ensure project root (containing utils/, artifacts, etc.) is on sys.path when
# invoked directly so that `from pipeline.phases.phase_three import main` works
# in environments where a user sets PYTHONPATH=.
import sys as _sys, os as _os
# ---------------- Thread / BLAS deterministic limits (set before numpy/torch import) ----------------
for _k, _v in {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}.items():
    # Do not overwrite if user explicitly set
    if _k not in _os.environ:
        _os.environ[_k] = _v
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _this_dir  # file already at root
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from pathlib import Path
import os
import json
import joblib
import numpy as np
import logging
import platform as _platform
from contextlib import nullcontext
import shutil
from datetime import datetime
import threading
import queue
import itertools
import time
import csv
import subprocess
import sys
import signal
import hashlib
import random
import json as _json_mod
from collections import deque

# Optional system metrics
try:
    import psutil as _psutil  # type: ignore
except Exception:
    _psutil = None
 
# ---------------- Store-and-forward disk buffer ----------------
class DiskRingBuffer:
    """Simple append-only JSONL ring buffer with size cap (bytes & records).

    Not crash-proof journal, but sufficient for transient broker outages.
    """
    def __init__(self, path: Path, max_bytes: int = 2_000_000, max_records: int = 5000):
        self.path = path
        self.max_bytes = max_bytes
        self.max_records = max_records
        self.lock = threading.Lock()
        self._count = 0
        # Initialize count lazily
        if self.path.exists():
            try:
                with self.path.open('r', encoding='utf-8') as f:
                    for _ in f:
                        self._count += 1
            except Exception:
                self._count = 0
    def append(self, obj: dict):
        line = _json_mod.dumps(obj, separators=(',', ':'))
        b = line.encode('utf-8')
        with self.lock:
            try:
                with self.path.open('ab') as f:
                    f.write(b + b"\n")
                self._count += 1
            except Exception:
                return
            # Enforce caps by truncation (rewrite newest tail)
            try:
                if (self.path.stat().st_size > self.max_bytes) or (self._count > self.max_records):
                    # Keep last N lines
                    keep = max(1, int(self.max_records * 0.8))
                    lines = self.path.read_text(encoding='utf-8', errors='ignore').splitlines()[-keep:]
                    self.path.write_text("\n".join(lines) + "\n", encoding='utf-8')
                    self._count = len(lines)
            except Exception:
                pass
    def pop_batch(self, limit: int = 50):
        with self.lock:
            if not self.path.exists():
                return []
            try:
                lines = self.path.read_text(encoding='utf-8').splitlines()
            except Exception:
                return []
            if not lines:
                return []
            batch = lines[:limit]
            remaining = lines[limit:]
            try:
                if remaining:
                    self.path.write_text("\n".join(remaining) + "\n", encoding='utf-8')
                else:
                    self.path.unlink()
                self._count = len(remaining)
            except Exception:
                pass
        out = []
        for ln in batch:
            try:
                out.append(_json_mod.loads(ln))
            except Exception:
                pass
        return out
    def size(self):
        with self.lock:
            return self._count


# Local reproducibility function
def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_all_seeds = set_seeds

# Metrics module not found; disabling metrics
_metrics_mod = None

# MQTT and inference libs (optional)
import paho.mqtt.client as mqtt

# Try ONNX runtime (preferred on edge). We'll add Raspberry Pi guidance if missing.
ORT = None
_onnx_import_error = None
try:  # pragma: no cover - environment dependent
    import onnxruntime as ort  # type: ignore
    ORT = ort
except Exception as _e:  # Capture to log explicit guidance later
    _onnx_import_error = _e
    ORT = None

# Try torch (fallback)
TORCH = None
try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    TORCH = torch
    try:
        # Enforce single-threaded execution to avoid BLAS oversubscription
        torch.set_num_threads(1)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:
    TORCH = None

# plotting for analyzer (used only by analysis tool)
import matplotlib
matplotlib.use("Agg")

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from jsonschema import validate as _json_validate, Draft7Validator as _Draft7Validator
from rotating_csv_logger import RotatingCSVLogger

# Import unified threshold manager
try:
    from threshold_manager import ThresholdManager
    THRESHOLD_MANAGER_AVAILABLE = True
except ImportError:
    ThresholdManager = None  # type: ignore
    THRESHOLD_MANAGER_AVAILABLE = False
# Make artifacts directory importable for the auto-generated inference wrapper
try:
    _artifacts_dir_guess = _os.path.join(_this_dir, "artifacts_phase2")
    if _os.path.isdir(_artifacts_dir_guess) and _artifacts_dir_guess not in _sys.path:
        _sys.path.insert(0, _artifacts_dir_guess)
    try:
        # Prefer local module import when path is on sys.path
        from inference_wrapper import EdgePredictor as _EdgePredictor  # type: ignore
    except Exception:
        # Fallback to package-style import if artifacts_phase2 is a package
        from artifacts_phase2.inference_wrapper import EdgePredictor as _EdgePredictor  # type: ignore
except Exception:
    _EdgePredictor = None
# Adaptive edge components
try:
    from edge.edge_buffer import get_edge_buffer  # type: ignore
    from edge.online_head import OnlineHead  # type: ignore
    from edge.drift_monitor import DriftMonitor  # type: ignore
    from edge.sample_uploader import SampleUploader  # type: ignore
except Exception:
    get_edge_buffer = None  # type: ignore
    OnlineHead = None       # type: ignore
    DriftMonitor = None     # type: ignore
    SampleUploader = None   # type: ignore

# ---------------- Config & paths ----------------
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PHASE2_DIR = BASE_DIR / "artifacts_phase2"
# New: dedicated directory for Phase 3 generated artifacts (logs, analysis, pointers, buffer)
PHASE3_DIR = BASE_DIR / "artifacts_phase3"
try:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Primary artifacts (Phase 2 outputs)
ONNX_MODEL_PATH = PHASE2_DIR / "model_hybrid.onnx"           # preferred for edge
FINAL_MODEL_PATH = PHASE2_DIR / "final_model_hybrid.pth"     # fallback
SCALER_PKL_PATH = PHASE2_DIR / "scaler.pkl"
SCALER_JSON_PATH = PHASE2_DIR / "scaler.json"
FEATURE_ORDER_PATH = PHASE2_DIR / "feature_order.json"
CLIP_BOUNDS_PATH = PHASE2_DIR / "clip_bounds.json"
EVAL_PATH = PHASE2_DIR / "evaluation.json"
LGBM_MODEL_PATH = PHASE2_DIR / "lgbm_model.pkl"               # optional ensemble

LOG_FILE_PATH = PHASE3_DIR / "phase3_predictions_log.csv"
ANALYSIS_ROLLING_PNG = PHASE3_DIR / "phase3_rolling_accuracy.png"
ANALYSIS_CM_PNG = PHASE3_DIR / "phase3_confusion_matrix.png"
ANALYSIS_SUMMARY_JSON = PHASE3_DIR / "phase3_analysis_summary.json"

# Dual-slot model pointer files
CURRENT_MODEL_POINTER = PHASE3_DIR / "current_model.txt"  # contains filename e.g. model_hybrid.onnx
PENDING_MODEL_POINTER = PHASE3_DIR / "pending_model.txt"  # orchestrator writes new filename here
PREVIOUS_MODEL_POINTER = PHASE3_DIR / "previous_model.txt"  # last successfully active model for rollback
_active_model_file = None  # track current active ONNX filename (string)

# Initialize disk buffer now that PHASE3_DIR is defined
BUFFER_PATH = PHASE3_DIR / "prediction_spool.jsonl"
_prediction_buffer = DiskRingBuffer(BUFFER_PATH)

# One-time migration: move existing Phase 3 artifacts from artifacts_phase2 -> artifacts_phase3
try:
    _migrate_list = [
        "phase3_predictions_log.csv",
        "phase3_rolling_accuracy.png",
        "phase3_confusion_matrix.png",
        "phase3_analysis_summary.json",
        "prediction_spool.jsonl",
        "current_model.txt",
        "pending_model.txt",
        "previous_model.txt",
    ]
    for _name in _migrate_list:
        _src = PHASE2_DIR / _name
        _dst = PHASE3_DIR / _name
        try:
            if _src.exists() and not _dst.exists():
                shutil.move(str(_src), str(_dst))
        except Exception:
            pass
except Exception:
    pass

# ---------------- Logging ----------------
LOG_FORMAT_ENV = os.getenv("LOG_FORMAT", "plain").lower()
if LOG_FORMAT_ENV == "json":
    class _JsonFormatter(logging.Formatter):
        def format(self, record):
            base = {
                "ts": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                base["exc"] = self.formatException(record.exc_info)
            return json.dumps(base)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    for _h in logging.getLogger().handlers:
        _h.setFormatter(_JsonFormatter())
logger = logging.getLogger("phase3")
logger.setLevel(logging.INFO)
# console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)
logger.propagate = False  # prevent duplicate logs via root
logger.info("Diagnostic: Logger '%s' has %d handlers", logger.name, len(logger.handlers))
_start_time_global = time.time()
_metrics_lock = threading.Lock()
_metrics = {"messages_total": 0,
            "inference_batches_total": 0,
            "last_batch_latency_ms": 0.0,
            # Count of MQTT authentication/authorization failures (rc in {4,5})
            "auth_failures_total": 0,
            # Latest processed batch size
            "last_batch_size": 0,
            # Rolling sum for avg batch size (divide by inference_batches_total)
            "total_batch_items": 0,
            # Latest observed queue fill ratios (0-1)
            "mqtt_queue_fill_ratio": 0.0,
            "proc_queue_fill_ratio": 0.0,
            # Queue backpressure metrics
            "queue_dropped_total": 0,
            "queue_size_max_observed": 0,
            "backpressure_events_total": 0,
            # Count of model switches (placeholder for future dual-slot logic)
            "model_switch_count": 0,
            # Processing lag seconds (avg of batch items now - enqueue ts)
            "last_batch_processing_lag_sec": 0.0,
            # Ingestion defense counters
            "rejected_payloads_total": 0,
            "rejected_payloads_size": 0,
            "rejected_payloads_feature_mismatch": 0,
            "rejected_payloads_json": 0,
            "rejected_payloads_topic": 0,
            "rejected_payloads_schema": 0,
            # Feature drift alerts
            "feature_drift_alerts_total": 0,
            # Logging resilience metrics (added for rotation & disk guard observability)
            "log_rotations_total": 0,
            "log_drops_low_disk_total": 0,
            "log_pruned_bytes_total": 0,
            # Coverage / preprocessing counters
            "skipped_low_coverage_total": 0,
            # Quant / parity metrics
            "onnx_parity_max_abs_delta": 0.0,
            "quant_parity_max_abs_delta": 0.0,
            "quantized_model_active": 0,
            # Edge adaptive extras
            "edge_samples_buffered_total": 0,
            "edge_samples_uploaded_total": 0,
            "edge_online_updates_total": 0,
            "edge_output_psi": 0.0,
            "edge_output_ks": 0.0}
            # Model lifecycle metrics
_metrics.update({
            "model_update_attempts_total": 0,
            "model_update_success_total": 0,
            "model_update_rejected_parity_total": 0,
            "model_rollback_total": 0,
            # Rolling latency percentiles and detection rate
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "latency_ms_p99": 0.0,
            "detection_rate_recent": 0.0,
            # Resource metrics
            "memory_rss_mb": 0.0,
            "cpu_percent": 0.0})

# ---------------- Checksum verification (fail-fast) ----------------
def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def _expected_onnx_sha256() -> str | None:
    """Return expected sha256 for ONNX model from metadata.json or artifact_manifest.json if available."""
    try:
        meta_path = PHASE2_DIR / "metadata.json"
        if meta_path.exists():
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
            for k in ("model_onnx_sha256", "model_onx_sha256", "model_sha256"):
                if isinstance(meta, dict) and meta.get(k):
                    return str(meta[k])
            am = meta.get("artifacts_manifest") or {}
            # support both dict-of-dicts and simple mapping
            ent = am.get("model_hybrid.onnx")
            if isinstance(ent, dict) and ent.get("sha256"):
                return str(ent["sha256"])
            if isinstance(am, dict) and isinstance(ent, str):
                return ent
    except Exception as e:
        logger.debug("metadata.json read failed: %s", e)
    try:
        man_path = PHASE2_DIR / "artifact_manifest.json"
        if man_path.exists():
            with man_path.open('r', encoding='utf-8') as f:
                mp = json.load(f)
            exp = mp.get("model_hybrid.onnx")
            if isinstance(exp, str) and len(exp) >= 32:
                return exp
    except Exception as e:
        logger.debug("artifact_manifest.json read failed: %s", e)
    return None

def _fail_if_model_checksum_mismatch(model_path: Path) -> None:
    """Compare actual vs expected sha256; exit(2) on mismatch. If expected missing and REQUIRE_MODEL_CHECKSUM=1, exit."""
    try:
        expected = _expected_onnx_sha256()
        if expected:
            actual = _file_sha256(model_path)
            if actual != expected:
                msg = f"Model checksum mismatch expected={expected[:12]} actual={actual[:12]} file={model_path.name}"
                logger.critical(msg)
                sys.exit(2)
        else:
            if os.getenv("REQUIRE_MODEL_CHECKSUM", "0").lower() in ("1","true","yes"):
                logger.critical("No expected model checksum found in metadata.json or artifact_manifest.json; refusing to start (REQUIRE_MODEL_CHECKSUM=1)")
                sys.exit(2)
            else:
                logger.warning("No expected ONNX checksum found (metadata/artifact_manifest); proceeding without verification")
    except SystemExit:
        raise
    except Exception as e:
        logger.warning("Checksum verification encountered an error: %s", e)

# Rolling windows for latency percentiles and detection rates
_latencies_ms_window = deque(maxlen=int(os.getenv("PHASE3_LATENCY_WINDOW", "2000")))
_det_rate_window = deque(maxlen=int(os.getenv("PHASE3_DET_RATE_WINDOW", "60")))

try:
    import http.server, socketserver
    class _MetricsHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            return
        def do_GET(self):
            if self.path == '/ready':
                # Consider ready when at least one inference batch OR uptime > 5s
                ready = False
                with _metrics_lock:
                    ready = (_metrics['inference_batches_total'] > 0) or (time.time()-_start_time_global > 5)
                data = ("ready" if ready else "starting").encode()
                code = 200 if ready else 503
                self.send_response(code)
                self.send_header('Content-Type','text/plain')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data); return
            if self.path == '/health':  # simple health probe (tests expect fields uptime_sec & disk_usage_mb)
                try:
                    total_bytes = 0
                    if PHASE2_DIR.exists():
                        for p in PHASE2_DIR.glob("**/*"):
                            if p.is_file():
                                try: total_bytes += p.stat().st_size
                                except Exception: pass
                    disk_usage_mb = round(total_bytes / (1024*1024), 3)
                except Exception:
                    disk_usage_mb = 0.0
                body = {"status": "ok", "uptime_sec": int(time.time()-_start_time_global), "disk_usage_mb": disk_usage_mb}
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data); return
            if self.path != '/metrics':
                self.send_response(404); self.end_headers(); return
            with _metrics_lock:
                lines = [
                    f"phase3_messages_total {_metrics['messages_total']}",
                    f"phase3_inference_batches_total {_metrics['inference_batches_total']}",
                    f"phase3_last_batch_latency_ms {_metrics['last_batch_latency_ms']}",
                    f"phase3_uptime_seconds {int(time.time()-_start_time_global)}",
                    f"phase3_queue_size {_async_logger.queue.qsize() if '_async_logger' in globals() else 0}",
                    f"phase3_auth_failures_total {_metrics['auth_failures_total']}",
                    f"phase3_last_batch_size {_metrics['last_batch_size']}",
                    f"phase3_avg_batch_size {(_metrics['total_batch_items'] / max(1,_metrics['inference_batches_total'])) if _metrics['inference_batches_total'] else 0}",
                    f"phase3_mqtt_queue_fill_ratio {_metrics['mqtt_queue_fill_ratio']}",
                    f"phase3_proc_queue_fill_ratio {_metrics['proc_queue_fill_ratio']}",
                    f"phase3_mqtt_queue_size {mqtt_queue.qsize()}",
                    f"phase3_proc_queue_size {proc_queue.qsize()}",
                    # Queue backpressure metrics
                    f"phase3_queue_dropped_total {_metrics.get('queue_dropped_total', 0)}",
                    f"phase3_queue_size_max_observed {_metrics.get('queue_size_max_observed', 0)}",
                    f"phase3_backpressure_events_total {_metrics.get('backpressure_events_total', 0)}",
                    f"phase3_mqtt_queue_max {MQTT_QUEUE_MAX}",
                    f"phase3_model_switch_count {_metrics['model_switch_count']}",
                    f"phase3_last_batch_processing_lag_sec {_metrics['last_batch_processing_lag_sec']}",
                    f"phase3_active_threshold {float(ACTIVE_THRESHOLD)}",
                    f"phase3_temperature {float(TEMPERATURE)}",
                    f"phase3_buffered_messages {_metrics.get('buffered_messages', 0)}",
                    f"phase3_rejected_payloads_total {_metrics['rejected_payloads_total']}",
                    f"phase3_rejected_payloads_size {_metrics['rejected_payloads_size']}",
                    f"phase3_rejected_payloads_feature_mismatch {_metrics['rejected_payloads_feature_mismatch']}",
                    f"phase3_rejected_payloads_json {_metrics['rejected_payloads_json']}",
                    f"phase3_rejected_payloads_schema {_metrics['rejected_payloads_schema']}",
                    f"phase3_rejected_payloads_topic {_metrics['rejected_payloads_topic']}",
                    f"phase3_feature_drift_alerts_total {_metrics.get('feature_drift_alerts_total',0)}",
                    f"phase3_log_rotations_total {_metrics.get('log_rotations_total',0)}",
                    f"phase3_log_drops_low_disk_total {_metrics.get('log_drops_low_disk_total',0)}",
                    f"phase3_log_pruned_bytes_total {_metrics.get('log_pruned_bytes_total',0)}",
                    f"phase3_skipped_low_coverage_total {_metrics.get('skipped_low_coverage_total',0)}",
                    f"phase3_alerts_count {len(_alert_manager.get_alerts())}",
                    f"phase3_onnx_parity_max_abs_delta {_metrics.get('onnx_parity_max_abs_delta',0.0)}",
                    f"phase3_quant_parity_max_abs_delta {_metrics.get('quant_parity_max_abs_delta',0.0)}",
                    f"phase3_quantized_model_active {_metrics.get('quantized_model_active',0)}",
                    f"phase3_edge_samples_buffered_total {_metrics.get('edge_samples_buffered_total',0)}",
                    f"phase3_edge_samples_uploaded_total {_metrics.get('edge_samples_uploaded_total',0)}",
                    f"phase3_edge_online_updates_total {_metrics.get('edge_online_updates_total',0)}",
                    f"phase3_edge_output_psi {_metrics.get('edge_output_psi',0.0)}",
                    f"phase3_edge_output_ks {_metrics.get('edge_output_ks',0.0)}",
                    f"phase3_model_update_attempts_total {_metrics.get('model_update_attempts_total',0)}",
                    f"phase3_model_update_success_total {_metrics.get('model_update_success_total',0)}",
                    f"phase3_model_update_rejected_parity_total {_metrics.get('model_update_rejected_parity_total',0)}",
                    f"phase3_model_rollback_total {_metrics.get('model_rollback_total',0)}",
                    f"phase3_latency_ms_p50 {_metrics.get('latency_ms_p50',0.0)}",
                    f"phase3_latency_ms_p95 {_metrics.get('latency_ms_p95',0.0)}",
                    f"phase3_latency_ms_p99 {_metrics.get('latency_ms_p99',0.0)}",
                    f"phase3_detection_rate_recent {_metrics.get('detection_rate_recent',0.0)}",
                    f"phase3_memory_rss_mb {_metrics.get('memory_rss_mb',0.0)}",
                    f"phase3_cpu_percent {_metrics.get('cpu_percent',0.0)}",
                ]
            data = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header('Content-Type','text/plain; version=0.0.4')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
    def _start_metrics_server():
        if not ENABLE_METRICS:
            return
        try:
            srv = socketserver.TCPServer(("0.0.0.0", METRICS_PORT), _MetricsHandler)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            logger.info("Metrics exporter running on :%d/metrics", METRICS_PORT)
        except Exception as e:
            logger.warning("Metrics server failed: %s", e)
except Exception:
    def _start_metrics_server():
        logger.warning("Metrics server unavailable (imports failed)")

# Load LGBM model if available
lgbm_model = None
if LGBM_MODEL_PATH.exists():
    try:
        lgbm_model = joblib.load(LGBM_MODEL_PATH)
        logger.info("Loaded LGBM model for ensemble.")
    except Exception as e:
        logger.warning("Failed to load LGBM model: %s", e)
        lgbm_model = None

from config import get_settings
settings = get_settings()
# Runtime config (env-overridable for secure deployments) now centralized in config.Settings
DEVICE = "cuda" if (TORCH and TORCH.cuda.is_available()) else "cpu"
MQTT_BROKER = settings.mqtt_broker
MQTT_PORT = settings.mqtt_port
MQTT_TOPIC = settings.mqtt_topic
HEALTH_TOPIC = settings.mqtt_health_topic
PREDICTIONS_TOPIC = settings.mqtt_predictions_topic
ALERT_THRESHOLD = settings.alert_threshold
MIN_FEATURE_COVERAGE = float(os.getenv("MIN_FEATURE_COVERAGE", 0.95))  # legacy retained
MIN_FEATURE_PRESENCE = float(os.getenv("MIN_FEATURE_PRESENCE", os.getenv("MIN_FEATURE_COVERAGE", 0.95)))
LOG_ROTATE_BYTES = settings.log_rotate_bytes
MQTT_TLS = settings.mqtt_tls
MQTT_USERNAME = settings.mqtt_username
MQTT_PASSWORD = settings.mqtt_password
MQTT_TLS_CA = settings.mqtt_tls_ca
MQTT_TLS_CERT = settings.mqtt_tls_cert
MQTT_TLS_KEY = settings.mqtt_tls_key
OMP_THREADS = settings.omp_threads
MICRO_BATCH_SIZE = settings.micro_batch_size
MICRO_BATCH_LATENCY_MS = settings.micro_batch_latency_ms
PROC_QUEUE_MAX = settings.proc_queue_max
HEALTH_INTERVAL = settings.health_interval
REQUIRE_HASH_MATCH = settings.require_hash_match
# Align with Phase 2 naming (artifact_manifest.json) and keep backward-compatible fallback
MANIFEST_PATH = PHASE2_DIR / "artifact_manifest.json"
MANIFEST_PATH_FALLBACK = PHASE2_DIR / "manifest_hashes.json"
METRICS_PORT = settings.metrics_port
ENABLE_METRICS = settings.enable_metrics
VALIDATE_INGRESS = settings.validate_ingress
INGRESS_SCHEMA_PATH = BASE_DIR / "schemas" / "ingress_payload.schema.json"
_INGRESS_SCHEMA = None
TOPIC_ALLOWLIST_RAW = os.getenv("MQTT_TOPIC_ALLOWLIST", "").strip()
TOPIC_ALLOWLIST = [t for t in (s.strip() for s in TOPIC_ALLOWLIST_RAW.split(",")) if t]

def security_preflight():
    """Security posture checks executed in STRICT_SECURITY mode (tests rely on failures).

    Enforced only if STRICT_SECURITY=1; otherwise returns True.
    Conditions:
      - If PRODUCTION=1 then TLS must be enabled.
      - If username set, password must also be set and sufficiently strong (>=12 chars, mixed complexity unless STRONG_PASSWORDS=0).
      - If METRICS_PUBLIC=1 then METRICS_TOKEN must be present.
      - If topic allowlist provided, primary MQTT_TOPIC must be in it.
    """
    strict_mode = os.getenv("STRICT_SECURITY", "0").lower() in ("1","true","yes")
    prod = os.getenv("PRODUCTION", "0").lower() in ("1","true","yes")
    enforce_tls = os.getenv("MQTT_ENFORCE_TLS", "0").lower() in ("1","true","yes")
    # Re-evaluate TLS each call (env may change in tests)
    # Evaluate TLS only from current env (avoid stale module constant so tests can toggle)
    tls = os.getenv("MQTT_TLS", "0").lower() in ("1","true","yes")
    if (prod or enforce_tls) and not tls:
        raise RuntimeError("TLS required in production or when MQTT_ENFORCE_TLS=1")
    # Username/password evaluated dynamically so tests that set env after first import still work
    dyn_username = os.getenv("MQTT_USERNAME", MQTT_USERNAME or "")
    dyn_password_env = os.getenv("MQTT_PASSWORD")
    dyn_password = dyn_password_env if dyn_password_env is not None else MQTT_PASSWORD
    if dyn_username and not dyn_password:
        raise RuntimeError("MQTT_PASSWORD required when MQTT_USERNAME set")
    if dyn_password:
        # Require strong passwords in any production/enforced TLS context OR if strict mode explicitly enabled.
        strong_required = (prod or enforce_tls or strict_mode) and os.getenv("STRONG_PASSWORDS", "1").lower() in ("1","true","yes")
        if strong_required:
            pw = dyn_password.strip()
            weak_list = {"password", "changeme", "admin", "administrator"}
            if len(pw) < 12 or pw.lower() in weak_list:
                raise RuntimeError("MQTT_PASSWORD not strong enough under STRONG_PASSWORDS policy")
    if strict_mode and os.getenv("METRICS_PUBLIC", "0").lower() in ("1","true","yes") and not os.getenv("METRICS_TOKEN"):
        raise RuntimeError("METRICS_PUBLIC=1 requires METRICS_TOKEN")
    # Topic allowlist enforcement (must happen in STRICT mode so tests see failure)
    if TOPIC_ALLOWLIST:
        primary_topic = os.getenv("MQTT_TOPIC", os.getenv("PRIMARY_TOPIC", "")) or globals().get("MQTT_TOPIC")
        if primary_topic and primary_topic not in TOPIC_ALLOWLIST:
            with _metrics_lock:
                _metrics['rejected_payloads_topic'] = _metrics.get('rejected_payloads_topic', 0) + 1
            raise RuntimeError("Primary MQTT_TOPIC not in allowlist")
    return True

def validate_onnx_parity(sample_count: int = 3) -> bool:
    """Best-effort parity check stub used by tests. Returns True if ONNX session exists or PyTorch model available.

    In a full implementation you would draw a few random feature vectors, run both runtimes and compare predictions.
    """
    if USE_ONNX and onnx_sess is not None:
        return True
    if TORCH is not None and pt_model is not None:
        return True
    return False

# ---------------- Parity & quantization gating -----------------
def _generate_random_inputs(n: int = 6, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n, NUM_FEATURES)).astype(np.float32)

def _run_backend_probs(x: np.ndarray, sess) -> np.ndarray:
    try:
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: x})[0]
        if out.ndim == 2 and out.shape[1] >= 2:
            # softmax class 1
            z = out - out.max(axis=1, keepdims=True)
            e = np.exp(z)
            return (e[:,1] / e.sum(axis=1)).astype(np.float32)
        # single logit => sigmoid
        flat = out.reshape(out.shape[0], -1)
        sig = 1.0/(1.0+np.exp(-flat[:,-1]))
        return sig.astype(np.float32)
    except Exception:
        return np.zeros((x.shape[0],), dtype=np.float32)

def evaluate_parity(base_sess, compare_sess, n: int = 8):
    """Return max absolute probability delta between two ONNX sessions."""
    x = _generate_random_inputs(n=n)
    p1 = _run_backend_probs(x, base_sess)
    p2 = _run_backend_probs(x, compare_sess)
    return float(np.max(np.abs(p1 - p2))) if p1.size and p2.size else 0.0

def maybe_activate_quantized_model():
    """If quantized ONNX present, evaluate parity vs current session and switch if within tolerance.

    Env controls:
      QUANT_PARITY_TOL=0.02 (max allowed abs prob delta)
      ENABLE_QUANT=1 to attempt activation
    """
    q_path = PHASE2_DIR / "model_hybrid_q.onnx"
    if not (USE_ONNX and onnx_sess is not None):
        return
    if not q_path.exists():
        return
    if os.getenv("ENABLE_QUANT", "0").lower() not in ("1","true","yes"):
        return
    tol = float(os.getenv("QUANT_PARITY_TOL", "0.02"))
    try:
        q_sess = ort.InferenceSession(str(q_path), providers=['CPUExecutionProvider'])
    except Exception as e:
        logger.warning("Quantized model load failed: %s", e)
        return
    delta = evaluate_parity(onnx_sess, q_sess, n=10)
    with _metrics_lock:
        _metrics['quant_parity_max_abs_delta'] = delta
    if delta <= tol:
        logger.info("Activating quantized ONNX model (delta=%.4f <= tol=%.4f)", delta, tol)
        globals()['onnx_sess'] = q_sess
        with _metrics_lock:
            _metrics['quantized_model_active'] = 1
    else:
        logger.warning("Quantized model parity delta=%.4f exceeds tol=%.4f; keeping original.", delta, tol)

try:
    maybe_activate_quantized_model()
except Exception:
    logger.debug("Quant gating failed", exc_info=True)

if VALIDATE_INGRESS:
    try:
        if INGRESS_SCHEMA_PATH.exists():
            import json as _json
            _INGRESS_SCHEMA = _json.loads(INGRESS_SCHEMA_PATH.read_text())
            _Draft7Validator.check_schema(_INGRESS_SCHEMA)
            logger.info("Ingress schema loaded for validation (%s)", INGRESS_SCHEMA_PATH.name)
        else:
            logger.warning("VALIDATE_INGRESS_SCHEMA=1 but schema file missing: %s", INGRESS_SCHEMA_PATH)
    except Exception as _e:
        logger.warning("Failed to initialize ingress schema validation: %s", _e)
        _INGRESS_SCHEMA = None

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def verify_artifact_hashes():
    manifest_path = MANIFEST_PATH if MANIFEST_PATH.exists() else (MANIFEST_PATH_FALLBACK if MANIFEST_PATH_FALLBACK.exists() else None)
    if not manifest_path:
        logger.warning("Manifest file not found (looked for %s and %s); skipping hash verification", MANIFEST_PATH.name, MANIFEST_PATH_FALLBACK.name)
        return True
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        logger.warning("Failed reading manifest %s: %s", manifest_path.name, e)
        return not REQUIRE_HASH_MATCH
    expected = manifest if isinstance(manifest, dict) else {}
    # keys we care about
    targets = {
        "model_hybrid.onnx": ONNX_MODEL_PATH,
        "final_model_hybrid.pth": FINAL_MODEL_PATH,
        "scaler.pkl": SCALER_PKL_PATH,
        "scaler.json": SCALER_JSON_PATH,
        "feature_order.json": FEATURE_ORDER_PATH,
    }
    all_ok = True
    for k, p in targets.items():
        if not p.exists():
            continue
        try:
            actual = _file_sha256(p)
            exp = expected.get(k)
            if exp and exp != actual:
                all_ok = False
                logger.error("Hash mismatch %s expected=%s actual=%s", k, exp[:12], actual[:12])
            else:
                logger.debug("Hash ok %s", k)
        except Exception as e:
            logger.warning("Hash compute failed for %s: %s", k, e)
    # Enforce strictness if either REQUIRE_HASH_MATCH is enabled or running in production
    strict_env = os.getenv("PRODUCTION", "0").lower() in ("1","true","yes")
    if not all_ok and (REQUIRE_HASH_MATCH or strict_env):
        raise RuntimeError("Artifact hash verification failed (strict mode or PRODUCTION=1)")
    return all_ok

verify_artifact_hashes()

# (Security preflight intentionally NOT executed at import time so tests can control when it's invoked.)
# Will be invoked in main() if STRICT_SECURITY=1 or PRODUCTION=1.

# ---------------- Dual-slot model loading helpers ----------------
def _read_pointer(path: Path):
    try:
        if path.exists():
            txt = path.read_text(encoding='utf-8').strip()
            return txt if txt else None
    except Exception:
        pass
    return None

def _write_pointer(path: Path, value: str):
    try:
        path.write_text(value + "\n", encoding='utf-8')
        return True
    except Exception as e:
        logger.warning("Failed writing pointer %s: %s", path.name, e)
        return False

def _warmup_onnx_session(model_path: Path):
    try:
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(str(model_path), sess_options, providers=['CPUExecutionProvider'])
        # warm-up inference
        dummy = np.zeros((1, NUM_FEATURES), dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        _ = sess.run(None, {input_name: dummy})
        return sess
    except Exception as e:
        logger.error("Warm-up failed for candidate model %s: %s", model_path.name, e)
        return None

def _maybe_switch_model():
    global onnx_sess, USE_ONNX, _active_model_file
    # read pending
    pending = _read_pointer(PENDING_MODEL_POINTER)
    if not pending:
        return
    if pending == _active_model_file:
        # clear stale pending
        try: PENDING_MODEL_POINTER.unlink()
        except Exception: pass
        return
    candidate = PHASE2_DIR / pending
    if not candidate.exists():
        logger.warning("Pending model file %s not found; ignoring", candidate)
        try: PENDING_MODEL_POINTER.unlink()
        except Exception: pass
        return
    with _metrics_lock:
        _metrics['model_update_attempts_total'] = _metrics.get('model_update_attempts_total',0)+1
    sess = _warmup_onnx_session(candidate)
    if sess is None:
        # leave pending for next attempt or external cleanup
        return
    # Optional parity gating vs current session
    parity_tol = float(os.getenv("MODEL_UPDATE_PARITY_TOL", os.getenv("QUANT_PARITY_TOL", "0.03")))
    parity_ok = True
    if os.getenv("MODEL_UPDATE_ENFORCE_PARITY", "1").lower() in ("1","true","yes") and onnx_sess is not None:
        try:
            delta = evaluate_parity(onnx_sess, sess, n=12)
            if delta > parity_tol:
                parity_ok = False
                logger.warning("Model update rejected: parity delta=%.4f > tol=%.4f", delta, parity_tol)
                with _metrics_lock:
                    _metrics['model_update_rejected_parity_total'] = _metrics.get('model_update_rejected_parity_total',0)+1
        except Exception as _pe:
            logger.warning("Parity evaluation failed; rejecting update by policy: %s", _pe)
            parity_ok = False
    if not parity_ok:
        try: PENDING_MODEL_POINTER.unlink()
        except Exception: pass
        return
    # successful warm-up: swap
    # Save previous pointer for rollback
    try:
        if _active_model_file:
            _write_pointer(PREVIOUS_MODEL_POINTER, _active_model_file)
    except Exception:
        pass
    onnx_sess = sess
    USE_ONNX = True
    prev = _active_model_file
    _active_model_file = candidate.name
    _write_pointer(CURRENT_MODEL_POINTER, _active_model_file)
    try: PENDING_MODEL_POINTER.unlink()
    except Exception: pass
    with _metrics_lock:
        _metrics['model_switch_count'] = _metrics.get('model_switch_count', 0) + 1
        _metrics['model_update_success_total'] = _metrics.get('model_update_success_total',0)+1
    logger.info("Model switch complete prev=%s new=%s", prev, _active_model_file)

    # Canary window monitoring (placeholder): could implement short-term performance checks here.

def rollback_model():
    """Rollback to previous model if pointer exists."""
    global onnx_sess, USE_ONNX, _active_model_file
    prev = _read_pointer(PREVIOUS_MODEL_POINTER)
    if not prev:
        logger.warning("Rollback requested but no previous model pointer present")
        return False
    candidate = PHASE2_DIR / prev
    if not candidate.exists():
        logger.warning("Rollback target %s missing", candidate)
        return False
    sess = _warmup_onnx_session(candidate)
    if sess is None:
        logger.warning("Rollback warmup failed for %s", candidate)
        return False
    onnx_sess = sess
    USE_ONNX = True
    _active_model_file = candidate.name
    _write_pointer(CURRENT_MODEL_POINTER, _active_model_file)
    with _metrics_lock:
        _metrics['model_rollback_total'] = _metrics.get('model_rollback_total',0)+1
    logger.info("Rollback successful -> %s", _active_model_file)
    return True

def _model_watch_loop():
    interval = int(os.getenv("MODEL_WATCH_INTERVAL", "10"))
    while not shutdown_event.is_set():
        try:
            _maybe_switch_model()
        except Exception:
            logger.debug("Model watch iteration failed", exc_info=True)
        shutdown_event.wait(interval)

# Optional signature verification (strict if ENFORCE_ARTIFACT_SIGNATURE=1)

# Artifact signature verification skipped (module not found)

# Logging level can be tuned via env (default INFO). Set VERBOSE_DIAGNOSTICS=1 to force DEBUG.
PHASE3_LOG_LEVEL = os.getenv("PHASE3_LOG_LEVEL", "INFO").upper()
if PHASE3_LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    PHASE3_LOG_LEVEL = "INFO"
try:
    logger.setLevel(getattr(logging, PHASE3_LOG_LEVEL))
except Exception:
    pass
if os.getenv("VERBOSE_DIAGNOSTICS", "0").lower() in ("1", "true", "yes"):
    logger.setLevel(logging.DEBUG)

# Alert logging controls
ALERT_LOGGING_ENABLED = os.getenv("ALERT_LOGGING_ENABLED", "1").lower() in ("1","true","yes")
ALERT_LOG_MAX_PER_SEC = int(os.getenv("ALERT_LOG_MAX_PER_SEC", "2"))  # max WARN lines per second
ALERT_LOG_SUMMARY_SEC = int(os.getenv("ALERT_LOG_SUMMARY_SEC", "10"))  # summarize suppressed alerts every N sec

# ---------------- Alerting webhook config ----------------
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()
ALERT_WEBHOOK_ENABLED = bool(ALERT_WEBHOOK_URL)
ALERT_EVAL_INTERVAL = int(os.getenv("ALERT_EVAL_INTERVAL", "15"))  # seconds
ALERT_LAG_THRESHOLD_SEC = float(os.getenv("ALERT_LAG_THRESHOLD_SEC", "5"))
ALERT_BUFFER_BACKLOG_THRESHOLD = int(os.getenv("ALERT_BUFFER_BACKLOG_THRESHOLD", "25"))
ALERT_AUTH_FAILURE_THRESHOLD = int(os.getenv("ALERT_AUTH_FAILURE_THRESHOLD", "3"))
ALERT_DRIFT_INCREMENT_THRESHOLD = int(os.getenv("ALERT_DRIFT_INCREMENT_THRESHOLD", "1"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "120"))
# Additional alert thresholds (optional)
ALERT_DETECTION_RATE_HIGH = float(os.getenv("ALERT_DETECTION_RATE_HIGH", "0.6"))  # trigger if recent detection rate too high
ALERT_CPU_PERCENT_HIGH = float(os.getenv("ALERT_CPU_PERCENT_HIGH", "85"))        # CPU percent high watermark
ALERT_MEM_RSS_MB_HIGH = float(os.getenv("ALERT_MEM_RSS_MB_HIGH", "500"))         # Memory RSS high watermark

_alert_last_sent = {}
_alert_lock = threading.Lock()

def _should_send_alert(key: str):
    now = time.time()
    with _alert_lock:
        last = _alert_last_sent.get(key, 0)
        if (now - last) >= ALERT_COOLDOWN_SEC:
            _alert_last_sent[key] = now
            return True
    return False

def _post_alert(event_type: str, details: dict):
    if not ALERT_WEBHOOK_ENABLED:
        return
    if not _should_send_alert(event_type):
        return
    body = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details,
        "service": "phase3_inference",
        "model_sha256": MODEL_ARTIFACT_SHA256,
    }
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(ALERT_WEBHOOK_URL, data=_json.dumps(body).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 (controlled URL)
            _ = resp.read()
        logger.info("Alert webhook sent type=%s", event_type)
    except Exception as e:
        logger.warning("Failed sending alert webhook type=%s err=%s", event_type, e)

def _alert_eval_loop():
    prev_auth_failures = 0
    prev_drift_alerts = 0
    while not shutdown_event.is_set():
        shutdown_event.wait(ALERT_EVAL_INTERVAL)
        if shutdown_event.is_set():
            break
        try:
            with _metrics_lock:
                lag = _metrics.get('last_batch_processing_lag_sec', 0.0)
                buf = _metrics.get('buffered_messages', 0)
                auth_fails = _metrics.get('auth_failures_total', 0)
                drift_alerts = _metrics.get('feature_drift_alerts_total', 0)
            if lag >= ALERT_LAG_THRESHOLD_SEC:
                _post_alert('high_processing_lag', {"lag_sec": lag, "threshold": ALERT_LAG_THRESHOLD_SEC})
            if buf >= ALERT_BUFFER_BACKLOG_THRESHOLD:
                _post_alert('publish_backlog', {"buffered": buf, "threshold": ALERT_BUFFER_BACKLOG_THRESHOLD})
            if (auth_fails - prev_auth_failures) >= ALERT_AUTH_FAILURE_THRESHOLD:
                _post_alert('auth_failures_spike', {"new_failures": auth_fails - prev_auth_failures, "total": auth_fails})
            prev_auth_failures = auth_fails
            if (drift_alerts - prev_drift_alerts) >= ALERT_DRIFT_INCREMENT_THRESHOLD:
                _post_alert('feature_drift', {"new_drift_alerts": drift_alerts - prev_drift_alerts, "total": drift_alerts})
            prev_drift_alerts = drift_alerts
            # Detection rate spike
            det_recent = _metrics.get('detection_rate_recent', 0.0)
            if det_recent >= ALERT_DETECTION_RATE_HIGH:
                _post_alert('detection_rate_spike', {"detection_rate_recent": det_recent, "threshold": ALERT_DETECTION_RATE_HIGH})
            # Resource usage alerts
            cpu = _metrics.get('cpu_percent', 0.0)
            mem = _metrics.get('memory_rss_mb', 0.0)
            if cpu >= ALERT_CPU_PERCENT_HIGH:
                _post_alert('cpu_usage_high', {"cpu_percent": cpu, "threshold": ALERT_CPU_PERCENT_HIGH})
            if mem >= ALERT_MEM_RSS_MB_HIGH:
                _post_alert('memory_usage_high', {"memory_rss_mb": mem, "threshold": ALERT_MEM_RSS_MB_HIGH})
        except Exception:
            logger.debug("Alert evaluation iteration failed", exc_info=True)

# Coverage warning toggle (set to 0 to suppress per-message coverage warnings)
COVERAGE_WARNINGS = os.getenv("COVERAGE_WARNINGS", "1").lower() in ("1","true","yes")

# ----------------- Load canonical feature order -----------------
def load_canonical_feature_order():
    """Load feature order from the canonical Phase 2 source with validation.
    
    Returns:
        tuple: (feature_order: List[str], num_features: int)
        
    Raises:
        FileNotFoundError: If Phase 2 artifacts not found
        RuntimeError: If feature order is invalid
    """
    if not PHASE2_DIR.exists():
        raise FileNotFoundError(f"Phase2 artifacts dir not found: {PHASE2_DIR}")

    if not FEATURE_ORDER_PATH.exists():
        raise FileNotFoundError(
            f"Canonical feature_order.json missing at {FEATURE_ORDER_PATH}. "
            "This file is created by Phase 2 (model training) and contains the "
            "authoritative feature order after any pruning. Please run Phase 2 first."
        )
    
    try:
        feature_order = json.loads(FEATURE_ORDER_PATH.read_text())
        if not isinstance(feature_order, list) or len(feature_order) == 0:
            raise RuntimeError("feature_order.json must contain a non-empty list of feature names")
        
        logger.info("Loaded canonical feature order: %d features from %s", 
                   len(feature_order), FEATURE_ORDER_PATH)
        return feature_order, len(feature_order)
        
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in feature_order.json at {FEATURE_ORDER_PATH}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load feature_order.json: {e}")

# Load canonical feature order with validation
FEATURE_ORDER, NUM_FEATURES = load_canonical_feature_order()

# clip bounds
if CLIP_BOUNDS_PATH.exists():
    try:
        cb = json.loads(CLIP_BOUNDS_PATH.read_text())
        CLIP_LOW = np.array(cb.get("p1", []), dtype=np.float32)
        CLIP_HIGH = np.array(cb.get("p99", []), dtype=np.float32)
        if len(CLIP_LOW) != NUM_FEATURES:
            logger.warning("clip_bounds length mismatch; ignoring clip bounds.")
            CLIP_LOW = CLIP_HIGH = None
    except Exception:
        CLIP_LOW = CLIP_HIGH = None
else:
    CLIP_LOW = CLIP_HIGH = None

# Load feature medians (fallback imputation) from Phase 1 stats if available
FEATURE_STATS_PATH = BASE_DIR / "artifacts_phase1" / "feature_stats.json"
FEATURE_MEDIANS = None
if FEATURE_STATS_PATH.exists():
    try:
        _fs = json.loads(FEATURE_STATS_PATH.read_text())
        med_list = []
        for feat in FEATURE_ORDER:
            entry = _fs.get(feat, {})
            # Prefer mean (already standardized) but treat as median fallback; if missing use 0.0
            med_list.append(float(entry.get("mean", 0.0)))
        FEATURE_MEDIANS = np.array(med_list, dtype=np.float32)
        logger.info("Loaded feature medians from feature_stats.json for imputation fallback.")
    except Exception as _fs_e:
        FEATURE_MEDIANS = None
        logger.debug("Failed loading feature_stats.json medians: %s", _fs_e)

# evaluation.json
eval_json = {}
if EVAL_PATH.exists():
    try:
        eval_json = json.loads(EVAL_PATH.read_text())
        logger.debug("Loaded evaluation.json keys=%s", list(eval_json.keys()))
    except Exception as e:
        logger.warning("Failed to parse evaluation.json: %s", e)
else:
    logger.warning("evaluation.json not found; using defaults")

# Load thresholds using unified threshold manager
if THRESHOLD_MANAGER_AVAILABLE:
    try:
        from threshold_manager import get_unified_thresholds
        unified = get_unified_thresholds()
        ACTIVE_THRESHOLD = unified['threshold']
        TEMPERATURE = unified['temperature']
        logger.info("Using unified threshold manager: threshold=%.4f, temperature=%.4f", ACTIVE_THRESHOLD, TEMPERATURE)
    except Exception as e:
        logger.warning("Threshold manager failed, falling back to evaluation.json: %s", e)
        # Fallback to original logic
        TEMPERATURE = float(eval_json.get("calibration", {}).get("temperature", 1.0))
        VAL_BEST_THRESHOLD = float(eval_json.get("meta", {}).get("val_best_threshold", eval_json.get("val", {}).get("threshold", ALERT_THRESHOLD)))
        VAL_BEST_THRESHOLD_DEEP_ONLY = float(eval_json.get("meta", {}).get("val_best_threshold_deep_only", VAL_BEST_THRESHOLD))
        ENSEMBLE_CFG = eval_json.get("ensemble", {"use_lgbm": False})
        ENABLE_ENSEMBLE = bool(ENSEMBLE_CFG.get("use_lgbm", False)) or os.getenv("ENABLE_ENSEMBLE", "0").lower() in ("1","true","yes")
        ACTIVE_THRESHOLD = VAL_BEST_THRESHOLD if ENABLE_ENSEMBLE else VAL_BEST_THRESHOLD_DEEP_ONLY
        logger.info("Active threshold=%.4f (ensemble=%s deep_only=%.4f full=%.4f)", ACTIVE_THRESHOLD, ENABLE_ENSEMBLE, VAL_BEST_THRESHOLD_DEEP_ONLY, VAL_BEST_THRESHOLD)
else:
    # Original threshold loading logic
    TEMPERATURE = float(eval_json.get("calibration", {}).get("temperature", 1.0))
    VAL_BEST_THRESHOLD = float(eval_json.get("meta", {}).get("val_best_threshold", eval_json.get("val", {}).get("threshold", ALERT_THRESHOLD)))
    VAL_BEST_THRESHOLD_DEEP_ONLY = float(eval_json.get("meta", {}).get("val_best_threshold_deep_only", VAL_BEST_THRESHOLD))
    ENSEMBLE_CFG = eval_json.get("ensemble", {"use_lgbm": False})
    ENABLE_ENSEMBLE = bool(ENSEMBLE_CFG.get("use_lgbm", False)) or os.getenv("ENABLE_ENSEMBLE", "0").lower() in ("1","true","yes")
    ACTIVE_THRESHOLD = VAL_BEST_THRESHOLD if ENABLE_ENSEMBLE else VAL_BEST_THRESHOLD_DEEP_ONLY
    logger.info("Active threshold=%.4f (ensemble=%s deep_only=%.4f full=%.4f)", ACTIVE_THRESHOLD, ENABLE_ENSEMBLE, VAL_BEST_THRESHOLD_DEEP_ONLY, VAL_BEST_THRESHOLD)

# Load other evaluation metadata
ENSEMBLE_CFG = eval_json.get("ensemble", {"use_lgbm": False})
KEPT_INDICES = eval_json.get("meta", {}).get("kept_feature_indices", None)
NUM_CLASSES = int(eval_json.get("meta", {}).get("num_classes", 2))
ALREADY_STANDARDIZED = eval_json.get("meta", {}).get("already_standardized", False)
logger.info("Diagnostic: ALREADY_STANDARDIZED from eval.json: %s", ALREADY_STANDARDIZED)
ENABLE_ENSEMBLE = bool(ENSEMBLE_CFG.get("use_lgbm", False)) or os.getenv("ENABLE_ENSEMBLE", "0").lower() in ("1","true","yes")

# if kept indices present, apply to FEATURE_ORDER
if KEPT_INDICES:
    try:
        FEATURE_ORDER = [FEATURE_ORDER[i] for i in KEPT_INDICES]
        NUM_FEATURES = len(FEATURE_ORDER)
        logger.debug("Applied kept_feature_indices; num_features=%d", NUM_FEATURES)
    except Exception:
        logger.warning("Failed to apply kept_feature_indices")

# ---------------- scaler loading (prefer JSON for edge) ----------------
scaler_json = None
if not SCALER_JSON_PATH.exists() and SCALER_PKL_PATH.exists():
    # One-time conversion path to enforce JSON-only at runtime
    try:
        scaler_obj = joblib.load(SCALER_PKL_PATH)
        mean = getattr(scaler_obj, 'mean_', None)
        scale = getattr(scaler_obj, 'scale_', None)
        if mean is not None and scale is not None:
            sjson = {
                "mean": np.asarray(mean, dtype=np.float32).tolist(),
                "scale": np.asarray(scale, dtype=np.float32).tolist(),
                "with_mean": bool(getattr(scaler_obj, 'with_mean', True)),
                "with_std": bool(getattr(scaler_obj, 'with_std', True)),
            }
            with open(SCALER_JSON_PATH, 'w') as jf:
                json.dump(sjson, jf)
            logger.info("Converted scaler.pkl -> scaler.json (canonical).")
        else:
            logger.warning("scaler.pkl missing mean_/scale_ attributes; proceeding without scaling")
    except Exception as e:
        logger.warning("Failed converting scaler.pkl to json: %s", e)

if SCALER_JSON_PATH.exists():
    try:
        scaler_json = json.loads(SCALER_JSON_PATH.read_text())
        logger.info("Loaded scaler.json (enforced).")
    except Exception as e:
        logger.error("Failed reading scaler.json (required): %s", e)
        scaler_json = None

SCALER_AVAILABLE = scaler_json is not None

# ---------------- ONNX vs PyTorch inference setup ----------------
USE_ONNX = False
onnx_sess = None
REQUIRE_ONNX = os.getenv("REQUIRE_ONNX", "0").lower() in ("1", "true", "yes")
_is_pi = _platform.machine().lower() in ("armv7l", "aarch64", "arm64")

def _log_onnx_guidance(reason: str):
    guidance = [
        f"ONNX runtime unavailable: {reason}",
        "Guidance:",
        "  1. On Raspberry Pi (ARM): install a matching aarch64 wheel, e.g.:",
        "     pip install --no-cache-dir onnxruntime==1.18.0",
        "     (Use 'python -V' to ensure wheel Python version compatibility.)",
        (
            "  2. If official wheel not provided for your Python version, either:\n"
            "     - downgrade Python (e.g. 3.11 -> 3.10) OR\n"
            "     - build from source: https://onnxruntime.ai/docs/build/"
        ),
        "  3. Set REQUIRE_ONNX=0 to allow PyTorch fallback (default).",
        "  4. Set REQUIRE_ONNX=1 to force a hard failure if ONNX missing (CI enforcement)."
    ]
    for line in guidance:
        logger.error(line) if REQUIRE_ONNX else logger.warning(line)

if ORT and ONNX_MODEL_PATH.exists():
    try:
        _fail_if_model_checksum_mismatch(ONNX_MODEL_PATH)
    except SystemExit:
        raise
    except Exception as _chk_err:
        logger.warning("Checksum preflight issue: %s", _chk_err)
    try:
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options, providers=['CPUExecutionProvider'])
        onnx_sess = sess
        USE_ONNX = True
        logger.info("Using ONNXRuntime (model=%s).", ONNX_MODEL_PATH.name)
        try:
            if not _read_pointer(CURRENT_MODEL_POINTER):
                _write_pointer(CURRENT_MODEL_POINTER, ONNX_MODEL_PATH.name)
            _active_model_file = _read_pointer(CURRENT_MODEL_POINTER) or ONNX_MODEL_PATH.name
        except Exception:
            _active_model_file = ONNX_MODEL_PATH.name
    except Exception as e:
        _log_onnx_guidance(f"initialization failed: {e}")
        onnx_sess = None
        USE_ONNX = False
elif ORT is None:
    # Provide Pi-specific guidance if running on ARM
    missing_reason = f"import error: {_onnx_import_error}" if _onnx_import_error else "module not installed"
    if _is_pi:
        _log_onnx_guidance(missing_reason + " (detected ARM platform)")
    else:
        logger.warning("ONNXRuntime not available (%s).", missing_reason)
    if REQUIRE_ONNX:
        raise RuntimeError("REQUIRE_ONNX=1 but onnxruntime not available")
else:
    # ORT present but ONNX model file missing
    logger.warning("ONNX model file %s not found; will attempt PyTorch fallback.", ONNX_MODEL_PATH)

# If no ONNX, prepare PyTorch model if available
pt_model = None
if not USE_ONNX:
    if TORCH is None:
        logger.error("No ONNX session and PyTorch not installed – cannot run inference.")
        raise RuntimeError("No inference runtime available (ONNX + PyTorch both unavailable).")
    if not FINAL_MODEL_PATH.exists():
        logger.error("PyTorch model not found at %s and ONNX not available.", FINAL_MODEL_PATH)
        raise FileNotFoundError("Model artifact missing.")
    # load model architecture dynamic stub: a minimal wrapper that matches saved state shape
    # We will load a simple Linear placeholder only to allow weights loading if needed.
    # Because your real model architecture is complex, prefer ONNX for edge. For PyTorch fallback, assume you have same class saved as state_dict.
    # For safety, we will attempt to import your DeepHybridModel from a local module if present, else fail gracefully.
    try:
        # Try to import model class from local module (Phase_Two or saved python file)
        from importlib import import_module
        mod = import_module("Phase_two_model_stub")  # optional: if user provides
        DeepHybridModel = getattr(mod, "DeepHybridModel")
        pt_model = DeepHybridModel(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
        state = TORCH.load(str(FINAL_MODEL_PATH), map_location=DEVICE)
        # support wrapped state dicts
        if isinstance(state, dict) and not all(isinstance(v, TORCH.Tensor) for v in state.values()):
            for k in ("state_dict", "model_state_dict", "weights"):
                if k in state:
                    state = state[k]
                    break
        pt_model.load_state_dict(state)
        pt_model.eval()
        logger.info("Loaded PyTorch model (DeepHybridModel).")
    except Exception:
        # Try to load state with a minimal model signature (best-effort) — if it fails, instruct user to use ONNX.
        try:
            # If user saved a full torchscript, try loading it as a script module
            scripted = TORCH.jit.load(str(FINAL_MODEL_PATH), map_location=DEVICE)
            pt_model = scripted
            pt_model.eval()
            logger.info("Loaded PyTorch scripted model from %s", FINAL_MODEL_PATH.name)
        except Exception as e:
            logger.error("Unable to load PyTorch model for inference: %s", e)
            raise RuntimeError("Provide ONNX model or ensure PyTorch model can be loaded (matching class or scripted).")

# Record baseline parity delta between ONNX and PyTorch if both available (non-fatal)
if USE_ONNX and onnx_sess is not None and pt_model is not None:
    try:
        base_inputs = np.random.default_rng(123).normal(0,1,size=(6, NUM_FEATURES)).astype(np.float32)
        # ONNX probs
        in_name = onnx_sess.get_inputs()[0].name
        onnx_out = onnx_sess.run(None, {in_name: base_inputs})[0]
        if onnx_out.ndim == 2 and onnx_out.shape[1] >= 2:
            z = onnx_out - onnx_out.max(axis=1, keepdims=True)
            e = np.exp(z); onnx_probs = e[:,1]/e.sum(axis=1)
        else:
            flat = onnx_out.reshape(onnx_out.shape[0], -1)
            onnx_probs = 1.0/(1.0+np.exp(-flat[:,-1]))
        with TORCH.no_grad():
            t_logits = pt_model(TORCH.tensor(base_inputs, dtype=TORCH.float32).to(DEVICE))
            if t_logits.ndim == 2 and t_logits.shape[1] >= 2:
                t_probs = TORCH.nn.functional.softmax(t_logits, dim=1).cpu().numpy()[:,1]
            else:
                flat = t_logits.detach().cpu().numpy().reshape(t_logits.shape[0], -1)
                t_probs = 1.0/(1.0+np.exp(-flat[:,-1]))
        delta = float(np.max(np.abs(onnx_probs - t_probs)))
        with _metrics_lock:
            _metrics['onnx_parity_max_abs_delta'] = delta
        logger.info("Baseline ONNX/PT parity max_abs_delta=%.5f", delta)
    except Exception:
        logger.debug("Baseline parity evaluation failed", exc_info=True)

# ----------------- Model artifact provenance hash -----------------
MODEL_ARTIFACT_SHA256 = "unknown"
try:
    _model_artifact_path = None
    if USE_ONNX and ONNX_MODEL_PATH.exists():
        try:
            _fail_if_model_checksum_mismatch(ONNX_MODEL_PATH)
        except SystemExit:
            raise
        except Exception:
            pass
        _model_artifact_path = ONNX_MODEL_PATH
    elif FINAL_MODEL_PATH.exists():
        _model_artifact_path = FINAL_MODEL_PATH
    if _model_artifact_path is not None:
        MODEL_ARTIFACT_SHA256 = _file_sha256(_model_artifact_path)
        logger.info("Model artifact hash computed (%s=%s)", _model_artifact_path.name, MODEL_ARTIFACT_SHA256[:16])
    else:
        logger.warning("No model artifact found to hash (both ONNX and PyTorch missing)")
except Exception as _mh_e:
    logger.warning("Failed computing model artifact hash: %s", _mh_e)

# ---------------- Label mapping helpers ----------------
LABEL_MAP = {"0": "Benign", "1": "Attack"}
LABEL_INDEX_FROM_NAME = {"benign": 0, "attack": 1}
try:
    lm = PHASE2_DIR / "label_mapping.json"
    if lm.exists():
        tmp = json.loads(lm.read_text())
        LABEL_MAP = tmp
        LABEL_INDEX_FROM_NAME = {str(v).strip().lower(): int(k) for k, v in tmp.items()}
except Exception:
    pass

def map_true_label(label):
    if label is None:
        return None
    try:
        iv = int(label)
        return iv if str(iv) in LABEL_MAP else None
    except Exception:
        s = str(label).strip().lower()
        return LABEL_INDEX_FROM_NAME.get(s, None)

# ----------------- Aliases & canonicalization -----------------
def _canon_key(k):
    return str(k).strip().lower().replace("-", "_").replace(" ", "_")

ALIAS_TO_CANON = {
    "src_port": "port_src", "sport": "port_src", "source_port": "port_src",
    "dst_port": "port_dst", "dport": "port_dst",
    "ttl": "ip_ttl",
    "length": "frame_len", "frame_length": "frame_len", "len": "frame_len", "pkt_len": "frame_len",
    "proto": "protocol", "protocol_name": "protocol"
}
ALIAS_GROUPS = {}
FEATURE_CANON_ORDER = [_canon_key(f) for f in FEATURE_ORDER]
for a, t in ALIAS_TO_CANON.items():
    ALIAS_GROUPS.setdefault(_canon_key(t), set()).add(_canon_key(a))

# Protocol coercion aligned with Phase 1 (ICMP=1, TCP=6, UDP=17, OTHER=0)
def _coerce_protocol_value(v):
    try:
        # numeric strings or numbers
        iv = int(v)
        return iv if iv in (1, 6, 17) else 0
    except Exception:
        s = str(v).strip()
        if s == "":
            return np.nan
        su = s.upper()
        # Try mapping artifact if present
        try:
            pm_path = PHASE2_DIR / "protocol_mapping.json"
            if pm_path.exists():
                pm = json.loads(pm_path.read_text())
                # keys could be names (e.g., "TCP")
                if su in pm:
                    return int(pm[su])
        except Exception:
            pass
        # Fallback fixed map
        if su in ("ICMP",):
            return 1
        if su in ("TCP",):
            return 6
        if su in ("UDP",):
            return 17
        return 0

# ----------------- Helpers -----------------
def to_float_safe(v):
    if v is None:
        return np.nan
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except Exception:
        s = str(v).strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return 1.0
        if s in ("false", "f", "no", "n", "0"):
            return 0.0
        return np.nan

def _is_probably_standardized(arr: np.ndarray) -> bool:
    z = arr[np.isfinite(arr)]
    if z.size == 0:
        return False
    frac_large = float(np.mean(np.abs(z) > 10.0))
    frac_small = float(np.mean(np.abs(z) <= 5.0))
    return frac_large < 0.05 and frac_small > 0.7

# prepare clip_mid if available
if CLIP_LOW is not None and CLIP_HIGH is not None:
    clip_mid = (CLIP_LOW + CLIP_HIGH) / 2.0
else:
    clip_mid = None

# scaler application (supports scaler.pkl object or scaler.json)
def apply_scaler_numpy(x_np: np.ndarray) -> np.ndarray:
    """x_np shape: (1, NUM_FEATURES). Requires scaler_json (JSON enforced)."""
    if not SCALER_AVAILABLE or scaler_json is None:
        return x_np
    mean = np.array(scaler_json.get("mean", [0.0]*NUM_FEATURES), dtype=np.float32)
    scale = np.array(scaler_json.get("scale", [1.0]*NUM_FEATURES), dtype=np.float32)
    return (x_np - mean) / (scale + 1e-12)

# ----------------- Preprocess -----------------
def preprocess_features(payload: dict):
    """Return numpy (1,NUM_FEATURES) or None if insufficient coverage"""
    src = payload.get("features", payload) if isinstance(payload, dict) else {}
    src_norm = {}
    if isinstance(src, dict):
        for k, v in src.items():
            src_norm[_canon_key(k)] = v

    vals = []
    missing = []
    for canon, feat_name in zip(FEATURE_CANON_ORDER, FEATURE_ORDER):
        v = src_norm.get(canon, None)
        if v is None:
            # try aliases
            for alias in ALIAS_GROUPS.get(canon, ()):
                if alias in src_norm:
                    v = src_norm[alias]; break
        # special protocol handling
        if canon == "protocol" and v is not None:
            v = _coerce_protocol_value(v)
        vn = to_float_safe(v)
        if vn is None or (isinstance(vn, float) and np.isnan(vn)):
            vals.append(np.nan)
            missing.append(feat_name)
        else:
            vals.append(float(vn))
    x = np.array(vals, dtype=np.float32).reshape(1, -1)
    # Replace inf/-inf with NaN for unified handling
    if np.isinf(x).any():
        try:
            with _metrics_lock:
                _metrics['sanitized_non_finite_total'] = _metrics.get('sanitized_non_finite_total', 0) + 1
        except Exception:
            pass
        x[~np.isfinite(x)] = np.nan
    present = int(np.sum(~np.isnan(x)))
    coverage = present / max(1, NUM_FEATURES)
    if coverage < MIN_FEATURE_PRESENCE:
        with _metrics_lock:
            _metrics["skipped_low_coverage_total"] = _metrics.get("skipped_low_coverage_total", 0) + 1
        if _metrics_mod is not None:
            try:
                _metrics_mod.set_feature_presence(coverage)
            except Exception:
                pass
        if COVERAGE_WARNINGS:
            logger.warning("Insufficient coverage %d/%d (%.1f%%). Missing sample first10=%s", present, NUM_FEATURES, coverage*100.0, missing[:10])
        else:
            logger.debug("Insufficient coverage %d/%d (%.1f%%)", present, NUM_FEATURES, coverage*100.0)
        return None
    # impute
    if np.isnan(x).any():
        if clip_mid is not None and len(clip_mid) == x.shape[1]:
            inds = np.where(np.isnan(x)); x[inds] = clip_mid[inds[1]]
        elif FEATURE_MEDIANS is not None and len(FEATURE_MEDIANS) == x.shape[1]:
            inds = np.where(np.isnan(x)); x[inds] = FEATURE_MEDIANS[inds[1]]
        else:
            x = np.nan_to_num(x, nan=0.0)
    # range clip if bounds present; otherwise at least clamp extreme outliers
    # detect if already standardized (env override has priority)
    env_std = os.getenv("INPUT_STANDARDIZED_HINT", "").strip().lower()
    if env_std in ("1", "true", "yes", "y"):
        std_hint = True
    elif env_std in ("0", "false", "no", "n"):
        std_hint = False
    else:
        std_hint = ALREADY_STANDARDIZED
    logger.debug("Diagnostic: std_hint=%s (ALREADY_STANDARDIZED=%s, env_raw=%s)", std_hint, ALREADY_STANDARDIZED, env_std)
    if not std_hint:
        # clip raw if bounds present
        if CLIP_LOW is not None and CLIP_HIGH is not None and len(CLIP_LOW)==x.shape[1]:
            x = np.clip(x, CLIP_LOW, CLIP_HIGH)
        if not SCALER_AVAILABLE:
            # One-time info to avoid flooding logs
            if not getattr(apply_scaler_numpy, "_warned_no_scaler", False):
                logger.info("No scaler available; proceeding with raw features (may degrade accuracy)")
                setattr(apply_scaler_numpy, "_warned_no_scaler", True)
            x_proc = x
        else:
            # apply scaler
            logger.debug("Diagnostic: Pre-scale x shape=%s, mean=%.4f, std=%.4f", x.shape, np.mean(x), np.std(x))
            try:
                x_proc = apply_scaler_numpy(x)
                logger.debug("Diagnostic: Post-scale x_proc shape=%s, mean=%.4f, std=%.4f", x_proc.shape, np.mean(x_proc), np.std(x_proc))
            except Exception as e:
                logger.exception("Scaler failed; abort inference: %s", e)
                return None
    else:
        logger.debug("Diagnostic: Skipping scaling (std_hint=true); x shape=%s, mean=%.4f, std=%.4f", x.shape, np.mean(x), np.std(x))
        x_proc = x  # assume already standardized
    # return numpy for ONNX or torch tensor for PT
    if _metrics_mod is not None:
        try:
            _metrics_mod.set_feature_presence(coverage)
        except Exception:
            pass
    return x_proc

# ----------------- Prediction -----------------
_edge_predictor = None
if _EdgePredictor is not None:
    try:
        _edge_predictor = _EdgePredictor(str(PHASE2_DIR))
        logger.info("EdgePredictor initialized (wrapper mode).")
    except Exception as _ep_e:
        logger.warning("EdgePredictor init failed; inline path retained: %s", _ep_e)

# -------- Adaptive components setup --------
ADAPTIVE_ENABLED = os.getenv("EDGE_ADAPTIVE_ENABLED", "1").lower() in ("1","true","yes")
ONLINE_CONFIDENCE_MARGIN = float(os.getenv("ONLINE_CONFIDENCE_MARGIN", "0.15"))  # distance from threshold to be considered confident
ONLINE_MAX_PROB = float(os.getenv("ONLINE_MAX_PROB", "0.999"))  # clamp extreme probs for stability
_edge_buffer = None
_online_head = None
_drift_monitor = None
_sample_uploader = None
if ADAPTIVE_ENABLED and get_edge_buffer is not None:
    try:
        _edge_buffer = get_edge_buffer()
        logger.info("EdgeBuffer initialized at %s", os.getenv("EDGE_BUFFER_PATH", "artifacts_phase2/edge_buffer.sqlite"))
    except Exception:
        logger.warning("EdgeBuffer init failed", exc_info=True)
if ADAPTIVE_ENABLED and OnlineHead is not None:
    try:
        _online_head = OnlineHead(NUM_FEATURES)
        if not _online_head.enabled:
            _online_head = None
        else:
            logger.info("OnlineHead enabled (incremental updates).")
    except Exception:
        logger.warning("OnlineHead init failed", exc_info=True)
def _set_drift_metrics(psi: float, ks: float):
    with _metrics_lock:
        _metrics['edge_output_psi'] = float(psi)
        _metrics['edge_output_ks'] = float(ks)
if ADAPTIVE_ENABLED and DriftMonitor is not None:
    try:
        _drift_monitor = DriftMonitor(_set_drift_metrics)
        if not _drift_monitor.enabled:
            _drift_monitor = None
        else:
            logger.info("DriftMonitor enabled (output distribution tracking).")
    except Exception:
        logger.warning("DriftMonitor init failed", exc_info=True)
if ADAPTIVE_ENABLED and SampleUploader is not None:
    try:
        # SampleUploader started later after MQTT client is ready; store factory args
        pass
    except Exception:
        pass

def predict_numpy(x_np: np.ndarray):
    """Return (pred_label:int, prob_attack:float) using EdgePredictor when available."""
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    # Preferred path
    if _edge_predictor is not None:
        try:
            res = _edge_predictor.predict_from_array(x_np, return_logits=False)
            return int(res['pred']), float(res['probs'][0])
        except Exception as e:
            logger.warning("EdgePredictor failed, falling back inline: %s", e)
    # Inline minimal ONNX fallback
    prob_attack = 0.0
    if USE_ONNX and onnx_sess is not None:
        try:
            input_name = onnx_sess.get_inputs()[0].name
            out = onnx_sess.run(None, {input_name: x_np.astype(np.float32)})[0]
            if out.ndim == 2 and out.shape[1] >= 2:
                z = out - out.max(axis=1, keepdims=True)
                e = np.exp(z)
                probs = e[:,1] / e.sum(axis=1)
                prob_attack = float(probs[0])
            else:
                flat = out.reshape(out.shape[0], -1)
                prob_attack = float(1.0/(1.0+np.exp(-flat[:,-1]))[0])
        except Exception as e:
            logger.warning("ONNX fallback failed: %s", e)
            prob_attack = 0.0
    # If still zero and torch available, attempt PT
    if prob_attack == 0.0 and TORCH is not None and pt_model is not None:
        try:
            tens = TORCH.tensor(x_np, dtype=TORCH.float32).to(DEVICE)
            with TORCH.no_grad():
                logits = pt_model(tens)
                if logits.ndim == 2 and logits.shape[1] >= 2:
                    probs = TORCH.nn.functional.softmax(logits / max(1e-6, float(TEMPERATURE)), dim=1).cpu().numpy()
                    prob_attack = float(probs[0,1])
                else:
                    arr = logits.detach().cpu().numpy().reshape(logits.shape[0], -1)
                    prob_attack = float(1.0/(1.0+np.exp(-arr[:,-1]))[0])
        except Exception as e:
            logger.warning("PyTorch fallback failed: %s", e)
    pred_thresh = int(prob_attack >= ACTIVE_THRESHOLD)
    # Adaptive hooks
    if ADAPTIVE_ENABLED:
        # Drift monitor observes probability regardless of confidence
        if _drift_monitor is not None:
            try: _drift_monitor.add(prob_attack)
            except Exception: pass
        # Store sample in EdgeBuffer (features keyed by original order)
        if _edge_buffer is not None:
            try:
                feats_rec = {FEATURE_ORDER[i]: float(x_np[0,i]) for i in range(min(len(FEATURE_ORDER), x_np.shape[1]))}
                _edge_buffer.add_sample(feats_rec, prob_attack, pred_thresh)
                with _metrics_lock:
                    _metrics['edge_samples_buffered_total'] = _metrics.get('edge_samples_buffered_total',0)+1
            except Exception:
                logger.debug("EdgeBuffer add_sample failed", exc_info=True)
        # Online head update only if confident
        if _online_head is not None:
            try:
                margin = abs(prob_attack - ACTIVE_THRESHOLD)
                if margin >= ONLINE_CONFIDENCE_MARGIN:
                    clamp_prob = max(1e-4, min(ONLINE_MAX_PROB, prob_attack))
                    pseudo_label = pred_thresh  # self-training
                    _online_head.partial_update(x_np[0], pseudo_label, sample_weight=1.0)
                    with _metrics_lock:
                        _metrics['edge_online_updates_total'] = _metrics.get('edge_online_updates_total',0)+1
            except Exception:
                logger.debug("OnlineHead update failed", exc_info=True)
    return pred_thresh, prob_attack

_async_logger = RotatingCSVLogger(LOG_FILE_PATH, metrics=_metrics, metrics_lock=_metrics_lock)

# ----------------- Alert manager -----------------
class AlertManager:
    def __init__(self, threshold=ACTIVE_THRESHOLD, max_alerts=500):
        self.threshold = threshold
        self.max_alerts = max_alerts
        self.lock = threading.Lock()
        self.alerts = []
        # rate limiting state
        self._bucket_sec = int(time.time())
        self._bucket_count = 0
        self._suppressed = 0
        self._last_summary = time.time()
    def check_and_record(self, timestamp, true_label, pred, prob_attack):
        if prob_attack >= self.threshold:
            alert = {"timestamp": timestamp, "true_label": true_label, "pred": pred, "prob_attack": float(prob_attack)}
            with self.lock:
                self.alerts.append(alert)
                if len(self.alerts) > self.max_alerts:
                    self.alerts = self.alerts[-self.max_alerts:]
            # rate-limited alert logging
            now = time.time()
            sec_now = int(now)
            with self.lock:
                if sec_now != self._bucket_sec:
                    self._bucket_sec = sec_now
                    self._bucket_count = 0
                do_log = ALERT_LOGGING_ENABLED and (self._bucket_count < max(0, ALERT_LOG_MAX_PER_SEC))
                if do_log:
                    self._bucket_count += 1
                else:
                    self._suppressed += 1
                # periodic summary of suppressed alerts
                if (now - self._last_summary) >= max(1, ALERT_LOG_SUMMARY_SEC) and self._suppressed > 0:
                    logger.warning("ALERTS suppressed: %d in last %d sec", self._suppressed, int(now - self._last_summary))
                    self._suppressed = 0
                    self._last_summary = now
            if do_log:
                logger.warning("ALERT: prob_attack=%.3f pred=%s at %s", prob_attack, pred, timestamp)
            return alert
        return None
    def get_alerts(self):
        with self.lock:
            return list(self.alerts)

_alert_manager = AlertManager(threshold=ACTIVE_THRESHOLD)

# ----------------- Metrics -----------------
total_msgs = 0
labeled_msgs = 0
correct_preds = 0
metrics_lock = threading.Lock()

# Bounded queues for async processing
MQTT_QUEUE_MAX = int(os.getenv("MQTT_QUEUE_MAX", "1000"))
WORKER_POOL_SIZE = max(1, int(os.getenv("WORKER_POOL_SIZE", str(max(1, os.cpu_count()//2)))))

# Raw MQTT message queue (bounded to prevent memory pressure)
mqtt_queue: "queue.Queue[mqtt.MQTTMessage]" = queue.Queue(maxsize=MQTT_QUEUE_MAX)
# Processed items queue (existing)
proc_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=PROC_QUEUE_MAX)
# Monotonic correlation/message id generator
_corr_counter = itertools.count(1)

# ----------------- Processing worker (micro-batch) ----------------
def _softmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    np.exp(arr, out=arr)
    sums = np.sum(arr, axis=1, keepdims=True)
    return arr / np.maximum(sums, 1e-12)

def _process_batch(items: list):
    # items: list of tuples (x_flat, true_idx, ts_iso, corr_id)
    global total_msgs, labeled_msgs, correct_preds
    if not items:
        return
    t_start = time.time()
    X = np.vstack([it[0].reshape(1, -1) for it in items]).astype(np.float32)
    n = X.shape[0]
    probs_deep = None
    used_pt_fallback = False
    # ONNX batch if available
    if USE_ONNX and onnx_sess is not None:
        try:
            input_name = onnx_sess.get_inputs()[0].name
            logits = onnx_sess.run(None, {input_name: X})[0].astype(np.float32)
            logits = logits / max(1e-6, float(TEMPERATURE))
            probs = _softmax(logits)
            probs_deep = probs[:, 1]
        except Exception as e:
            logger.warning("ONNX batch inference failed: %s; falling back to per-item", e)
            probs_deep = None
    if probs_deep is None:
        # Per-item fallback (PyTorch or per-item ONNX)
        probs = []
        for i in range(n):
            try:
                pred_i, prob_i = predict_numpy(X[i:i+1])
                probs.append(prob_i)
                used_pt_fallback = True
            except Exception as e:
                logger.warning("Inference failed for item %d: %s", i, e)
                probs.append(0.0)
        probs_deep = np.array(probs, dtype=np.float32)
    # LGBM ensemble if configured
    probs_final = probs_deep.copy()
    if lgbm_model is not None and ENSEMBLE_CFG.get("use_lgbm", False):
        try:
            p_lgb = lgbm_model.predict_proba(X)[:, 1]
            w = float(ENSEMBLE_CFG.get("weight_deep", 0.5))
            probs_final = w * probs_deep + (1.0 - w) * p_lgb
        except Exception as e:
            logger.warning("LGBM ensemble batch failed: %s; using deep probs", e)
            probs_final = probs_deep

    preds = (probs_final >= float(ACTIVE_THRESHOLD)).astype(int)

    # --- Feature drift detection (rolling window z-score monitoring) ---
    # Initialize global rolling stats holders lazily
    global _drift_window, _drift_mean, _drift_m2, _drift_count
    try:
        _ = _drift_count  # type: ignore
    except NameError:
        _drift_window = int(os.getenv("DRIFT_WINDOW", "500"))
        _drift_mean = np.zeros(X.shape[1], dtype=np.float64)
        _drift_m2 = np.zeros(X.shape[1], dtype=np.float64)  # sum of squared diffs
        _drift_count = 0
    # Update running mean/variance (Welford) using batch
    for row in X:
        _drift_count += 1
        delta = row - _drift_mean
        _drift_mean += delta / _drift_count
        delta2 = row - _drift_mean
        _drift_m2 += delta * delta2
        # Optional decay to bound influence (simple exponential decay)
        if _drift_count > _drift_window:
            decay = float(os.getenv("DRIFT_DECAY", "0.002"))
            _drift_mean = (1 - decay) * _drift_mean + decay * row
            # Lightly decay m2 as well
            _drift_m2 *= (1 - decay)
    # Compute per-feature std (with ddof=1 if count>1)
    denom = max(1, _drift_count - 1)
    drift_std = np.sqrt(np.maximum(_drift_m2 / denom, 1e-12))
    # Compare against training stats if available (scaler_json mean/scale else eval_json meta)
    drift_alert = False
    try:
        if scaler_json is not None:
            ref_mean = np.array(scaler_json.get("mean", [0.0]*X.shape[1]), dtype=np.float64)
            ref_scale = np.array(scaler_json.get("scale", [1.0]*X.shape[1]), dtype=np.float64)
        else:
            ref_mean = np.zeros(X.shape[1], dtype=np.float64)
            ref_scale = np.ones(X.shape[1], dtype=np.float64)
        # Compute absolute z of rolling mean relative to training distribution per feature
        z = np.abs((_drift_mean - ref_mean) / (ref_scale + 1e-6))
        # Drift if any feature mean deviates strongly; threshold env tunable
        z_thresh = float(os.getenv("DRIFT_Z_THRESHOLD", "6.0"))
        frac_thresh = float(os.getenv("DRIFT_FRACTION_THRESHOLD", "0.1"))
        frac_exceed = float(np.mean(z > z_thresh))
        if frac_exceed >= frac_thresh and _drift_count > 50:  # require some burn-in
            drift_alert = True
    except Exception:
        pass
    if drift_alert:
        with _metrics_lock:
            _metrics['feature_drift_alerts_total'] = _metrics.get('feature_drift_alerts_total', 0) + 1
        if ALERT_LOGGING_ENABLED:
            logger.warning("FEATURE DRIFT ALERT: frac_exceed=%.3f z_thresh=%.1f count=%d", frac_exceed, z_thresh, _drift_count)
    # Update metrics and emit outputs
    labeled = 0
    correct = 0
    for i, (x_flat, true_idx, ts, corr_id) in enumerate(items):
        pred = int(preds[i])
        prob = float(probs_final[i])
        # CSV log
        _async_logger.log(ts, true_idx if true_idx is not None else "", pred, round(prob, 6))
        # Alerts
        _alert_manager.check_and_record(ts, true_idx, pred, prob)
        # Metrics
        if true_idx is not None:
            labeled += 1
            if int(true_idx) == pred:
                correct += 1
        # Publish prediction
        try:
            out = {
                "timestamp": ts,
                "schema_version": "1.0",
                "model_version": str(os.getenv("MODEL_VERSION", "unknown")),
                "model_sha256": MODEL_ARTIFACT_SHA256,
                "correlation_id": int(corr_id),
                "pred": int(pred),
                "prediction": int(pred),  # alias for dashboard
                "prob_attack": float(prob),
                "threshold": float(ACTIVE_THRESHOLD),
                "temperature": float(TEMPERATURE),
                "ensemble_enabled": bool(ENABLE_ENSEMBLE),
                # Approximate per-item inference latency (batch latency / batch size)
                "inference_ms": float(((time.time() - t_start) * 1000.0) / max(1, len(items))),
            }
            if true_idx is not None:
                out["true_label"] = int(true_idx)
            res = client.publish(PREDICTIONS_TOPIC, json.dumps(out), qos=0)
            try:
                # If publish result indicates failure, buffer it
                if getattr(res, 'rc', 0) != 0:
                    _prediction_buffer.append({"topic": PREDICTIONS_TOPIC, "payload": out})
            except Exception:
                pass
        except Exception:
            # On any exception, buffer the payload for retry
            try:
                _prediction_buffer.append({"topic": PREDICTIONS_TOPIC, "payload": out})  # type: ignore
            except Exception:
                pass
    with metrics_lock:
        total_msgs += n
        labeled_msgs += labeled
        correct_preds += correct
    # metrics exporter counters
    try:
        # Compute processing lag (avg now - enqueue timestamp)
        now_ts = time.time()
        lags = []
        for (_, _, ts_iso, _) in items:
            try:
                lags.append(now_ts - datetime.fromisoformat(ts_iso).timestamp())
            except Exception:
                pass
        avg_lag = float(sum(lags)/len(lags)) if lags else 0.0
        with _metrics_lock:
            _metrics['messages_total'] += n
            _metrics['inference_batches_total'] += 1
            batch_ms = (time.time() - t_start) * 1000.0
            _metrics['last_batch_latency_ms'] = batch_ms
            _metrics['last_batch_size'] = n
            _metrics['total_batch_items'] += n
            _metrics['last_batch_processing_lag_sec'] = avg_lag
            try:
                _metrics['mqtt_queue_fill_ratio'] = mqtt_queue.qsize() / float(max(1, mqtt_queue.maxsize))
                _metrics['proc_queue_fill_ratio'] = proc_queue.qsize() / float(max(1, proc_queue.maxsize))
            except Exception:
                pass
            _metrics['buffered_messages'] = _prediction_buffer.size()
            # Update rolling windows
            try:
                # Use per-item approx latency (batch_ms/n) for each item
                per_item = float(batch_ms / max(1, n))
                for _ in range(n):
                    _latencies_ms_window.append(per_item)
                if _latencies_ms_window:
                    arr = np.array(_latencies_ms_window, dtype=np.float32)
                    _metrics['latency_ms_p50'] = float(np.percentile(arr, 50))
                    _metrics['latency_ms_p95'] = float(np.percentile(arr, 95))
                    _metrics['latency_ms_p99'] = float(np.percentile(arr, 99))
            except Exception:
                pass
            try:
                # Detection rate = fraction of alerts in last window of seconds
                # Append current batch detection ratio
                det_ratio = float(np.mean((probs_final >= float(ACTIVE_THRESHOLD)).astype(np.float32)))
                _det_rate_window.append(det_ratio)
                if _det_rate_window:
                    _metrics['detection_rate_recent'] = float(np.mean(_det_rate_window))
            except Exception:
                pass
            # Resource metrics via psutil if available
            try:
                if _psutil is not None:
                    p = _psutil.Process()
                    _metrics['memory_rss_mb'] = float(p.memory_info().rss / (1024*1024))
                    # cpu_percent with interval=0 returns last computed; it's okay for rough telemetry
                    _metrics['cpu_percent'] = float(p.cpu_percent(interval=0.0))
            except Exception:
                pass
    except Exception:
        pass
    # Prometheus histogram
    if _metrics_mod:
        _metrics_mod.observe_latency(time.time() - t_start)
        try:
            _metrics_mod.set_threshold(float(ACTIVE_THRESHOLD))
            _metrics_mod.set_temperature(float(TEMPERATURE))
        except Exception:
            pass

def _mqtt_message_worker():
    """Worker thread that processes raw MQTT messages from mqtt_queue."""
    while not shutdown_event.is_set():
        try:
            msg = mqtt_queue.get(timeout=0.1)
            try:
                _process_mqtt_message(msg)
            except Exception as e:
                logger.exception("Failed processing MQTT message: %s", e)
            finally:
                mqtt_queue.task_done()
        except queue.Empty:
            continue

def _process_mqtt_message(msg):
    """Process a single MQTT message (moved from on_message callback)."""
    try:
        # Topic allowlist enforcement
        if TOPIC_ALLOWLIST and msg.topic not in TOPIC_ALLOWLIST:
            with _metrics_lock:
                _metrics['rejected_payloads_total'] += 1
                _metrics['rejected_payloads_topic'] = _metrics.get('rejected_payloads_topic', 0) + 1
            return
        payload_raw = msg.payload.decode(errors='ignore')
        # Size guard (bytes)
        max_bytes = int(os.getenv("INGEST_MAX_BYTES", "65536"))
        if len(payload_raw) > max_bytes:
            with _metrics_lock:
                _metrics['rejected_payloads_total'] += 1
                _metrics['rejected_payloads_size'] += 1
            return
        try:
            payload = json.loads(payload_raw)
        except Exception:
            with _metrics_lock:
                _metrics['rejected_payloads_total'] += 1
                _metrics['rejected_payloads_json'] += 1
            logger.debug("Invalid JSON payload rejected (size=%d)", len(payload_raw))
            return
        if VALIDATE_INGRESS and _INGRESS_SCHEMA is not None:
            try:
                _json_validate(instance=payload, schema=_INGRESS_SCHEMA)
            except Exception as _ve:
                with _metrics_lock:
                    _metrics['rejected_payloads_total'] += 1
                    _metrics['rejected_payloads_schema'] = _metrics.get('rejected_payloads_schema', 0) + 1
                logger.debug("Ingress schema reject: %s", _ve)
                return
        true_raw = payload.get("label", None)
        true_idx = map_true_label(true_raw)
        features = payload.get("features", payload)
        # Feature count guard
        if isinstance(features, (list, tuple)) and len(features) != NUM_FEATURES:
            with _metrics_lock:
                _metrics['rejected_payloads_total'] += 1
                _metrics['rejected_payloads_feature_mismatch'] += 1
            return
        ts_now_iso = datetime.utcnow().isoformat()
        x_np = preprocess_features(features)
        if x_np is None:
            _async_logger.log(ts_now_iso, true_idx if true_idx is not None else "", "", 0.0)
            return
        corr_id = next(_corr_counter)
        item = (x_np.reshape(-1), true_idx, ts_now_iso, corr_id)
        try:
            proc_queue.put_nowait(item)
        except queue.Full:
            try:
                _ = proc_queue.get_nowait()
                proc_queue.put_nowait(item)
            except Exception:
                pass
    except Exception:
        logger.exception("Failed processing MQTT message")

def _process_worker():
    """Worker thread that processes preprocessed items from proc_queue."""
    batch = []
    last = time.time()
    while not shutdown_event.is_set():
        try:
            item = proc_queue.get(timeout=0.05)
            batch.append(item)
            # Trigger on size
            if len(batch) >= max(1, int(MICRO_BATCH_SIZE)):
                _process_batch(batch)
                batch = []
                last = time.time()
        except queue.Empty:
            pass
        # Trigger on latency
        if batch and ((time.time() - last) * 1000.0 >= float(MICRO_BATCH_LATENCY_MS)):
            _process_batch(batch)
            batch = []
            last = time.time()
    # Flush remaining
    if batch:
        _process_batch(batch)

# ----------------- MQTT client & callbacks ----------------
client = mqtt.Client(protocol=mqtt.MQTTv5, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
# Flow control: avoid unbounded memory usage under burst
try:
    client.max_inflight_messages_set(20)
    client.max_queued_messages_set(1000)
except Exception:
    pass

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker %s:%d", MQTT_BROKER, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        logger.info("Subscribed to %s", MQTT_TOPIC)
        # Security posture warning: in production (ENV=prod) enforce TLS guidance
        try:
            if os.getenv("ENV", "dev").lower() == "prod" and not MQTT_TLS:
                logger.warning("SECURITY: Running in ENV=prod without MQTT_TLS enabled. Set MQTT_TLS=1 and provide certs (MQTT_TLS_CA/MQTT_TLS_CERT/MQTT_TLS_KEY) for encryption & auth.")
        except Exception:
            pass
    else:
        logger.error("MQTT connect failed rc=%s", rc)
        # Track auth/authorization failures per MQTT v5 return codes (4=bad user/pass, 5=not authorized)
        if rc in (4,5):
            try:
                with _metrics_lock:
                    _metrics['auth_failures_total'] += 1
            except Exception:
                pass

def on_message(client, userdata, msg, properties=None):
    """Lightweight MQTT callback - just enqueue raw messages for worker processing."""
    # Update queue metrics
    current_size = mqtt_queue.qsize()
    with _metrics_lock:
        _metrics['mqtt_queue_fill_ratio'] = current_size / float(max(1, mqtt_queue.maxsize))
        _metrics['queue_size_max_observed'] = max(_metrics.get('queue_size_max_observed', 0), current_size)
        
        # Check for backpressure threshold (90% capacity)
        if current_size >= (0.9 * mqtt_queue.maxsize):
            _metrics['backpressure_events_total'] += 1
            if _metrics['backpressure_events_total'] % 50 == 1:  # Log every 50th event
                logger.warning("BACKPRESSURE: queue utilization high (%d/%d = %.1f%%)", 
                             current_size, mqtt_queue.maxsize, 
                             (current_size / mqtt_queue.maxsize) * 100)
    
    try:
        mqtt_queue.put_nowait(msg)
        with _metrics_lock:
            _metrics['messages_total'] += 1
    except queue.Full:
        # Drop message if queue full to prevent backlog/memory pressure
        with _metrics_lock:
            _metrics['rejected_payloads_total'] += 1
            _metrics['queue_dropped_total'] += 1
            
        if _metrics['queue_dropped_total'] % 100 == 1:  # Log every 100th drop
            logger.warning("MQTT queue full -> dropped %d messages (latest topic=%s)", 
                         _metrics['queue_dropped_total'], msg.topic)

client.on_connect = on_connect
client.on_message = on_message

# Exponential backoff connect helper
def mqtt_connect_with_backoff():
    backoff_initial = float(os.getenv("MQTT_BACKOFF_INITIAL", 1.0))
    backoff_max = float(os.getenv("MQTT_BACKOFF_MAX", 30.0))
    backoff_mult = float(os.getenv("MQTT_BACKOFF_MULT", 2.0))
    jitter_frac = float(os.getenv("MQTT_BACKOFF_JITTER", 0.1))
    max_attempts_env = os.getenv("MQTT_BACKOFF_MAX_ATTEMPTS", "0")
    try:
        max_attempts = int(max_attempts_env)
    except Exception:
        max_attempts = 0
    attempt = 0
    delay = backoff_initial
    while not shutdown_event.is_set():
        attempt += 1
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            logger.info("MQTT connected on attempt %d", attempt)
            return True
        except Exception as e:
            logger.warning("MQTT connect attempt %d failed: %s", attempt, e)
            if max_attempts and attempt >= max_attempts:
                logger.error("Exceeded max MQTT connection attempts (%d)", max_attempts)
                return False
            sleep_for = delay
            if jitter_frac > 0:
                j = sleep_for * jitter_frac
                sleep_for = max(0.1, sleep_for + random.uniform(-j, j))
            logger.info("Retrying MQTT connect in %.2fs (attempt=%d)", sleep_for, attempt)
            time.sleep(sleep_for)
            delay = min(delay * backoff_mult, backoff_max)
    return False

shutdown_event = threading.Event()

# Retry buffered predictions loop
def _buffer_retry_loop():
    while not shutdown_event.is_set():
        try:
            batch = _prediction_buffer.pop_batch(limit=25)
            if not batch:
                shutdown_event.wait(2.0)
                continue
            for entry in batch:
                try:
                    topic = entry.get('topic', PREDICTIONS_TOPIC)
                    payload = entry.get('payload')
                    if payload is None:
                        continue
                    res = client.publish(topic, json.dumps(payload), qos=0)
                    if getattr(res, 'rc', 0) != 0:
                        _prediction_buffer.append(entry)
                except Exception:
                    try:
                        _prediction_buffer.append(entry)
                    except Exception:
                        pass
            with _metrics_lock:
                _metrics['buffered_messages'] = _prediction_buffer.size()
        except Exception:
            time.sleep(2.0)

# ----------------- Analyzer & analysis helpers ----------------
def parse_csv_lenient(path):
    try:
        df = None
        if Path(path).exists():
            df = __import__("pandas").read_csv(path)
            # coerce fields
            if "true_label" in df.columns:
                def _coerce(v):
                    try: return int(v)
                    except: 
                        try:
                            return map_true_label(v)
                        except:
                            return None
                df["true_label_idx"] = df["true_label"].apply(_coerce)
            else:
                df["true_label_idx"] = None
            if "pred" in df.columns:
                df["pred"] = __import__("pandas").to_numeric(df["pred"], errors="coerce").astype("Int64")
            if "prob_attack" in df.columns:
                df["prob_attack"] = __import__("pandas").to_numeric(df["prob_attack"], errors="coerce")
        return df
    except Exception as e:
        logger.warning("parse_csv_lenient failed: %s", e)
        return None

def analyze_logs(window=2000):
    df = parse_csv_lenient(LOG_FILE_PATH)
    summary = {"total": 0, "with_labels": 0, "overall_accuracy": 0.0}
    if df is None or df.empty:
        Path(ANALYSIS_SUMMARY_JSON).write_text(json.dumps(summary, indent=2))
        return
    df_valid = df.dropna(subset=["pred"])
    summary["total"] = int(len(df_valid))
    df_labeled = df_valid.dropna(subset=["true_label_idx"])
    summary["with_labels"] = int(len(df_labeled))
    if len(df_labeled)>0:
        acc = float((df_labeled["true_label_idx"].astype(int) == df_labeled["pred"].astype(int)).mean())
        summary["overall_accuracy"] = acc
        # rolling plot
        try:
            roll = (df_labeled["true_label_idx"].astype(int) == df_labeled["pred"].astype(int)).rolling(window=window, min_periods=1).mean()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,3))
            plt.plot(roll.values)
            plt.ylim(0,1)
            plt.title(f"Rolling accuracy (window={window})")
            plt.tight_layout(); plt.savefig(ANALYSIS_ROLLING_PNG); plt.close()
        except Exception:
            logger.warning("rolling plot failed")
        # confusion matrix
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(df_labeled["true_label_idx"].astype(int), df_labeled["pred"].astype(int), labels=[0,1])
            import matplotlib.pyplot as plt
            disp = ConfusionMatrixDisplay(cm, display_labels=[LABEL_MAP.get("0","Benign"), LABEL_MAP.get("1","Attack")])
            disp.plot(values_format="d")
            plt.tight_layout(); plt.savefig(ANALYSIS_CM_PNG); plt.close()
        except Exception:
            logger.warning("confusion matrix plotting failed")
    Path(ANALYSIS_SUMMARY_JSON).write_text(json.dumps(summary, indent=2))

def _write_and_launch_detached_analyzer():
    try:
        runner_path = PHASE2_DIR / "phase3_analyzer_runner.py"
        runner_code = f"""#!/usr/bin/env python3
import json, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LOG_FILE = Path(r"{LOG_FILE_PATH}")
OUT_SUM = Path(r"{ANALYSIS_SUMMARY_JSON}")
OUT_ROLL = Path(r"{ANALYSIS_ROLLING_PNG}")
OUT_CM = Path(r"{ANALYSIS_CM_PNG}")

def main():
    if not LOG_FILE.exists():
        OUT_SUM.write_text(json.dumps({{"total":0,"with_labels":0,"overall_accuracy":0.0}}, indent=2)); return
    df = pd.read_csv(LOG_FILE)
    if 'true_label' in df.columns:
        def _c(v):
            try: return int(v)
            except: return None
        df['true_label_idx'] = df['true_label'].apply(_c)
    dfv = df.dropna(subset=['pred'])
    dfl = dfv.dropna(subset=['true_label_idx'])
    summary = {{"total": int(len(dfv)), "with_labels": int(len(dfl)), "overall_accuracy": 0.0}}
    if len(dfl)>0:
        acc = float((dfl['true_label_idx'].astype(int) == dfl['pred'].astype(int)).mean())
        summary['overall_accuracy'] = acc
        roll = (dfl['true_label_idx'].astype(int) == dfl['pred'].astype(int)).rolling(window=2000, min_periods=1).mean()
        plt.figure(figsize=(8,3)); plt.plot(roll.values); plt.ylim(0,1); plt.tight_layout(); plt.savefig(OUT_ROLL); plt.close()
        try:
            cm = confusion_matrix(dfl['true_label_idx'].astype(int), dfl['pred'].astype(int), labels=[0,1])
            disp = ConfusionMatrixDisplay(cm, display_labels=[\"Benign\",\"Attack\"])
            disp.plot(values_format='d'); plt.tight_layout(); plt.savefig(OUT_CM); plt.close()
        except Exception:
            pass
    OUT_SUM.write_text(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
"""
        runner_path.write_text(runner_code)
        if os.name == "nt":
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen([sys.executable, str(runner_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
        else:
            subprocess.Popen([sys.executable, str(runner_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
        return True
    except Exception as e:
        logger.warning("Failed to launch detached analyzer: %s", e)
        return False

# --------------- Graceful shutdown & health (single main implementation) ---------------
def _graceful_shutdown(signum, frame):
    logger.info("Shutdown signal (%s) received; stopping...", signum)
    shutdown_event.set()
    try:
        client.unsubscribe(MQTT_TOPIC)
        client.loop_stop()
        client.disconnect()
    except Exception:
        pass
    try:
        # Prefer close(); fallback to stop() for backward compat if reloaded stale module
        try:
            getattr(_async_logger, 'close')( )
        except AttributeError:
            getattr(_async_logger, 'stop')()
    except Exception as e:
        logger.warning("Async logger stop failed: %s", e)
    try:
        launched = _write_and_launch_detached_analyzer()
        if not launched:
            analyze_logs()
    except Exception as e:
        logger.exception("Analyzer failed: %s", e)

signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)

def _health_loop():
    interval = max(3, HEALTH_INTERVAL)
    while not shutdown_event.is_set():
        start = time.time()
        try:
            qsize = _async_logger.queue.qsize()
            alerts = len(_alert_manager.get_alerts())
            with metrics_lock:
                lm = labeled_msgs
                tp = correct_preds
            logger.debug("Health: queue=%d alerts=%d labeled=%d correct=%d", qsize, alerts, lm, tp)
            payload = {
                "ts": datetime.utcnow().isoformat(),
                "queue": qsize,
                "alerts": alerts,
                "labeled_msgs": lm,
                "correct_preds": tp,
                "uptime_sec": int(time.time() - _start_time_global),
                "temperature": float(TEMPERATURE),
                "threshold": float(ACTIVE_THRESHOLD),
                "ensemble_enabled": bool(ENABLE_ENSEMBLE),
                "min_feature_presence": float(MIN_FEATURE_PRESENCE),
                "model_sha256": MODEL_ARTIFACT_SHA256,
                "last_correlation_id": (next(_corr_counter) - 1),
                "last_batch_latency_ms": _metrics.get('last_batch_latency_ms', 0.0),
                "last_batch_size": _metrics.get('last_batch_size', 0),
                "skipped_low_coverage_total": _metrics.get('skipped_low_coverage_total', 0),
            }
            try:
                client.publish(HEALTH_TOPIC, json.dumps(payload), qos=0)
            except Exception:
                pass
        except Exception:
            logger.debug("Health loop iteration failed", exc_info=True)
        elapsed = time.time() - start
        to_sleep = interval - elapsed
        if to_sleep > 0:
            shutdown_event.wait(to_sleep)

_health_thread = threading.Thread(target=_health_loop, daemon=True)
_health_thread.start()

try:
    _SEED = int(os.getenv('PIPELINE_GLOBAL_SEED', '1337'))
    set_all_seeds(_SEED)
    logger.info("Phase 3 seeds set (seed=%d)", _SEED)
except Exception as _se:
    logger.warning("Failed to set seeds: %s", _se)

def main():
    logger.info("Starting Phase 3 listener (broker=%s:%d topic=%s). Using runtime=%s", MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, "ONNX" if USE_ONNX else "PyTorch")
    try:
        if os.getenv("STRICT_SECURITY", "0").lower() in ("1","true","yes") or os.getenv("PRODUCTION", "0").lower() in ("1","true","yes"):
            security_preflight()
            logger.info("Security preflight passed (strict=%s production=%s)", os.getenv("STRICT_SECURITY"), os.getenv("PRODUCTION"))
    except Exception as sec_e:
        logger.error("Security preflight failed: %s", sec_e)
        return
    _start_metrics_server()
    try:
        os.environ["OMP_NUM_THREADS"] = str(OMP_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(OMP_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(OMP_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(OMP_THREADS)
    except Exception:
        pass
    # Start MQTT message worker pool
    for i in range(WORKER_POOL_SIZE):
        t = threading.Thread(target=_mqtt_message_worker, daemon=True, name=f"mqtt-worker-{i}")
        t.start()
        logger.info("Started MQTT message worker %d/%d", i+1, WORKER_POOL_SIZE)
    
    # Start other workers
    threading.Thread(target=_process_worker, daemon=True).start()
    threading.Thread(target=_buffer_retry_loop, daemon=True).start()
    threading.Thread(target=_model_watch_loop, daemon=True).start()
    if ALERT_WEBHOOK_ENABLED:
        threading.Thread(target=_alert_eval_loop, daemon=True).start()
        logger.info("Alert webhook evaluation loop started (url=%s)", ALERT_WEBHOOK_URL)
    try:
        if MQTT_USERNAME:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
        if MQTT_TLS:
            try:
                if MQTT_TLS_CA or MQTT_TLS_CERT:
                    client.tls_set(ca_certs=MQTT_TLS_CA, certfile=MQTT_TLS_CERT, keyfile=MQTT_TLS_KEY)
                else:
                    client.tls_set()
                client.tls_insecure_set(False)
                logger.info("TLS enabled for MQTT connection")
            except Exception as e:
                logger.warning("Failed to configure TLS: %s", e)
        if not mqtt_connect_with_backoff():
            logger.error("MQTT connection failed after backoff attempts; exiting")
            return
        client.loop_start()
    except Exception:
        logger.exception("Failed connecting to MQTT broker")
        raise
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        logger.info("Main loop exiting; performing shutdown tasks")
        _graceful_shutdown(None, None)

if __name__ == "__main__":
    main()
