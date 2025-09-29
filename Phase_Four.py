#!/usr/bin/env python3
# Phase_4.py
"""
Phase 4 - Edge Deployment Simulation (quantization, ONNX export, benchmarking, MQTT subscriber)

Improvements:
 - robust_torch_load helper used for safer torch.load behaviour
 - env metadata added to benchmark JSON (python version, platform, torch version, onnxruntime version if available)
 - cleaned CLI, TLS/auth options for MQTT, prediction/health topics support
 - robustly handles state_dict / module / quantized checkpoint variants
"""
from __future__ import annotations

import os
# ---------------- Deterministic thread limits (set before heavy libs import) ----------------
for _k, _v in {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}.items():
    if _k not in os.environ:  # honor user override
        os.environ[_k] = _v
import sys
import json
import time
import argparse
import logging
import hashlib
import signal
import http.server
import socketserver
import threading
import random
from pathlib import Path
from rotating_csv_logger import RotatingCSVLogger
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
import platform as _platform
from datetime import datetime
from contextlib import nullcontext

import joblib
import pickle
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
try:
    torch.set_num_threads(1)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(1)
except Exception:
    pass

# Optional packages
try:
    import psutil
except Exception:
    psutil = None

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore
    _onnx_import_error = None
except Exception as _onnx_e:
    onnx = None
    ort = None
    _onnx_import_error = _onnx_e
_metrics_mod = None
_phase_verify_or_exit = None

# ----------------------
# Paths & defaults
# ----------------------

BASE_DIR = Path(__file__).resolve().parent
PHASE1_DIR = BASE_DIR / "artifacts_phase1"
PHASE2_DIR = BASE_DIR / "artifacts_phase2"
# New: dedicated directory for Phase 4 generated artifacts
PHASE4_DIR = BASE_DIR / "artifacts_phase4"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)


# Inputs (Phase 1/2 artifacts)
BEST_MODEL_PATH = PHASE2_DIR / "best_model_hybrid.pth"
FINAL_MODEL_PATH = PHASE2_DIR / "final_model_hybrid.pth"
QUANT_MODEL_PATH = PHASE2_DIR / "quantized_model_hybrid.pth"
ONNX_MODEL_PATH  = PHASE2_DIR / "model_hybrid.onnx"
QUANT_REPORT_JSON = PHASE2_DIR / "quantization_report.json"
PHASE2_SCALER = PHASE2_DIR / "scaler.pkl"      # prefer phase2, fallback to phase1
PHASE1_DATA_NPZ = PHASE1_DIR / "data.npz"
PHASE1_DATA_PKL = PHASE1_DIR / "data.pkl"
PHASE2_FEATURE_ORDER = PHASE2_DIR / "feature_order.json"
PHASE2_CLIP_BOUNDS = PHASE2_DIR / "clip_bounds.json"

# Outputs (Phase 4 artifacts)
DEFAULT_LOG_CSV = PHASE4_DIR / "phase4_predictions_log.csv"
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10000000"))  # 10 MB default
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
MIN_FREE_DISK_MB = int(os.getenv("MIN_FREE_DISK_MB", "50"))
BENCHMARK_OUTPUT = PHASE4_DIR / "phase4_benchmark.json"

# MQTT defaults (unified across phases via env overrides)
MQTT_PREDICTIONS_TOPIC_DEFAULT = os.getenv("MQTT_PREDICTIONS_TOPIC", "iot/traffic/predictions")
MQTT_HEALTH_TOPIC_DEFAULT = os.getenv("MQTT_HEALTH_TOPIC", "iot/traffic/health")
HEALTH_INTERVAL_DEFAULT = 15  # seconds

# ----------------------
# Hyper / env
# ----------------------
SEED = 42
set_all_seeds(SEED)

# ----------------------
# Logging (plain or JSON via LOG_FORMAT=json)
# ----------------------
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
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase4")

# ----------------------
# Model definition (must match Phase_2)
# ----------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        w = w.unsqueeze(2)
        return x * w

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class DeepHybridModel(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super().__init__()
        cnn_channels = 32
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2)
        )
        self.resblock1 = ResidualBlock1D(cnn_channels, dropout=0.2)
        self.se1 = SEBlock(cnn_channels)
        self.resblock2 = ResidualBlock1D(cnn_channels, dropout=0.2)
        self.se2 = SEBlock(cnn_channels)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(cnn_channels, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = SelfAttentionBlock(embed_dim=64, num_heads=2)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.fc1 = nn.Linear(64 + 64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x_seq = x.unsqueeze(1)          # (B, 1, F)
        x_seq = self.cnn_stem(x_seq)    # (B, C, F//2)
        x_seq = self.resblock1(x_seq)
        x_seq = self.se1(x_seq)
        x_seq = self.resblock2(x_seq)
        x_seq = self.se2(x_seq)
        x_seq = self.pool(x_seq)        # (B, C, F//4)
        x_seq = x_seq.permute(0, 2, 1)  # (B, F//4, C)
        lstm_out, _ = self.lstm(x_seq)  # (B, F//4, 64)
        attn_out = self.attn(lstm_out)  # (B, F//4, 64)
        x_seq = attn_out.mean(dim=1)    # (B, 64)
        x_mlp = self.mlp(x)             # (B, 64)
        x_cat = torch.cat([x_seq, x_mlp], dim=1)
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        return self.fc2(x_cat)

# ----------------------
# Robust torch.load helper
# ----------------------
def robust_torch_load(path, map_location=None):
    """
    Load torch checkpoint robustly. Accepts state_dict, full module, or legacy formats.
    """
    # prefer weights_only=False to allow arbitrary objects; catch TypeError for older torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # last-resort: open file and attempt pickle.load
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            raise e

# ----------------------
# Helpers: load artifacts & preprocessing
# ----------------------
def find_feature_order(explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    if os.path.exists(PHASE2_FEATURE_ORDER):
        return PHASE2_FEATURE_ORDER
    alt = os.path.join(PHASE1_DIR, "feature_order.json")
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError("feature_order.json not found (searched override, artifacts_phase2, artifacts_phase1)")

class DummyScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.array(mean, dtype=np.float32)
        self.scale_ = np.array(scale, dtype=np.float32)

    def transform(self, X):
        return (X - self.mean_) / (self.scale_ + 1e-12)

def load_artifacts(feature_order_override=None, scaler_override=None):
    """Load feature ordering, clip bounds, and scaler (pkl or json) from artifacts.

    Returns
    -------
    (feature_order, clip_low, clip_high, scaler)
    """
    feat_path = find_feature_order(feature_order_override)
    with open(feat_path, "r") as f:
        feature_order = json.load(f)

    # Clip bounds (prefer phase2, fallback phase1)
    clip_path = PHASE2_CLIP_BOUNDS if os.path.exists(PHASE2_CLIP_BOUNDS) else os.path.join(PHASE1_DIR, "clip_bounds.json")
    if not os.path.exists(clip_path):
        raise FileNotFoundError("clip_bounds.json missing in artifacts")
    with open(clip_path, "r") as f:
        clip_bounds = json.load(f)
    clip_low = np.array(clip_bounds["p1"], dtype=np.float32)
    clip_high = np.array(clip_bounds["p99"], dtype=np.float32)

    # Scaler loading strategy (canonical JSON):
    # 1. Prefer scaler.json (phase2 then phase1)
    # 2. Else try scaler.pkl (phase2 then phase1) and immediately emit scaler.json for future runs
    scaler = None
    json_candidates = []
    if scaler_override and scaler_override.endswith('.json'):
        json_candidates.append(scaler_override)
    json_candidates.extend([
        os.path.join(PHASE2_DIR, "scaler.json"),
        os.path.join(PHASE1_DIR, "scaler.json"),
    ])
    for jp in json_candidates:
        if os.path.exists(jp):
            try:
                with open(jp, "r") as f:
                    sj = json.load(f)
                if isinstance(sj, dict) and "mean" in sj and "scale" in sj:
                    scaler = DummyScaler(sj["mean"], sj["scale"])
                    logger.info("Loaded DummyScaler from %s", jp)
                    break
            except Exception as e:
                logger.warning("Failed reading %s: %s", jp, e)

    if scaler is None:
        pkl_candidates = []
        if scaler_override and scaler_override.endswith('.pkl'):
            pkl_candidates.append(scaler_override)
        pkl_candidates.extend([
            os.path.join(PHASE2_DIR, "scaler.pkl"),
            os.path.join(PHASE1_DIR, "scaler.pkl"),
        ])
        for p in pkl_candidates:
            if os.path.exists(p):
                try:
                    scaler_obj = joblib.load(p)
                    # Attempt to extract mean_/scale_ and emit canonical JSON
                    mean = getattr(scaler_obj, 'mean_', None)
                    scale = getattr(scaler_obj, 'scale_', None)
                    if mean is not None and scale is not None:
                        scaler_json = {
                            "mean": np.asarray(mean, dtype=np.float32).tolist(),
                            "scale": np.asarray(scale, dtype=np.float32).tolist(),
                            "with_mean": bool(getattr(scaler_obj, 'with_mean', True)),
                            "with_std": bool(getattr(scaler_obj, 'with_std', True)),
                        }
                        out_json_path = os.path.join(PHASE2_DIR, 'scaler.json') if 'artifacts_phase2' in p else os.path.join(PHASE1_DIR, 'scaler.json')
                        try:
                            with open(out_json_path, 'w') as jf:
                                json.dump(scaler_json, jf)
                            logger.info("Emitted canonical scaler.json to %s", out_json_path)
                        except Exception as je:
                            logger.warning("Failed emitting scaler.json: %s", je)
                        scaler = DummyScaler(scaler_json["mean"], scaler_json["scale"])
                    else:
                        logger.warning("Scaler pickle missing mean_/scale_; cannot canonicalize to JSON")
                        scaler = scaler_obj
                    logger.info("Loaded scaler from %s", p)
                    break
                except Exception as e:
                    logger.warning("Failed loading %s: %s", p, e)

    if scaler is None:
        raise FileNotFoundError("No scaler artifact found (scaler.json or scaler.pkl)")

    return feature_order, clip_low, clip_high, scaler

def load_test_data():
    """Load test data from Phase 1 artifacts for benchmarking."""
    data = None
    
    # Try .npz first
    if os.path.exists(PHASE1_DATA_NPZ):
        try:
            npz_data = np.load(PHASE1_DATA_NPZ, allow_pickle=True)
            data = {
                "X_test": npz_data["X_test"],
                "y_test": npz_data["y_test"]
            }
            logger.info("Loaded test data from .npz format")
        except Exception as e:
            logger.warning("Failed to load .npz: %s", e)
    
    # Fallback to .pkl
    if data is None and os.path.exists(PHASE1_DATA_PKL):
        try:
            with open(PHASE1_DATA_PKL, "rb") as f:
                pkl_data = pickle.load(f)
                data = {
                    "X_test": pkl_data["X_test"],
                    "y_test": pkl_data["y_test"]
                }
            logger.info("Loaded test data from .pkl format")
        except Exception as e:
            logger.warning("Failed to load .pkl: %s", e)
    
    if data is None:
        raise FileNotFoundError(f"No test data found in {PHASE1_DATA_NPZ} or {PHASE1_DATA_PKL}")
    
    return data["X_test"], data["y_test"]

# ----------------------
# Quantization + ONNX export
# ----------------------
def quantize_and_save(orig_model_path=BEST_MODEL_PATH, quant_path=QUANT_MODEL_PATH, device="cpu"):
    # If a prior quantization gating report exists and failed, surface and skip.
    try:
        if QUANT_REPORT_JSON.exists():
            with open(QUANT_REPORT_JSON, 'r') as f:
                qr = json.load(f)
            if not qr.get('passed', False):
                logger.warning("Existing quantization_report.json indicates gating failure; skipping new quantization attempt.")
                return None
            else:
                logger.info("Existing quantization gating PASSED (auc_drop=%.4f acc_drop=%.4f)",
                            -qr.get('deltas',{}).get('roc_auc_drop',0.0), -qr.get('deltas',{}).get('thr_accuracy_drop',0.0))
    except Exception as e:
        logger.debug("Could not read prior quantization report: %s", e)
    logger.info("Quantize: Loading original model state...")
    if not os.path.exists(orig_model_path):
        raise FileNotFoundError(orig_model_path)
    feature_order, _, _, _ = load_artifacts()
    num_features = len(feature_order)

    model = DeepHybridModel(num_features=num_features)
    loaded = None
    try:
        sd = robust_torch_load(orig_model_path, map_location=device)
        # If it's a state_dict-like mapping
        if isinstance(sd, dict):
            try:
                model.load_state_dict(sd)
                loaded = "state_dict"
                logger.info("Loaded state_dict into model.")
            except Exception:
                for k in ("state_dict", "model_state_dict", "weights"):
                    if k in sd and isinstance(sd[k], dict):
                        model.load_state_dict(sd[k])
                        loaded = k
                        logger.info("Loaded nested '%s' state_dict into model.", k)
                        break
    except Exception as e:
        logger.debug("state_dict load attempt failed/raised: %s", e)

    if loaded is None:
        try:
            obj = robust_torch_load(orig_model_path, map_location=device)
            if isinstance(obj, nn.Module):
                model = obj
                loaded = "module_object"
                logger.info("Loaded full module object from checkpoint.")
            else:
                if isinstance(obj, dict):
                    raise RuntimeError("Loaded dict could not be applied to model.")
                else:
                    raise RuntimeError("Checkpoint is of unexpected type: %s" % type(obj))
        except Exception as e:
            logger.exception("Failed to interpret checkpoint as state_dict or module: %s", e)
            raise

    model.eval()
    logger.info("Applying dynamic quantization (Linear only, skip LSTM for better accuracy) ...")
    q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    logger.info("Saving quantized model object to %s", quant_path)
    torch.save(q_model, quant_path)
    # Derive a sidecar state_dict filename safely
    if hasattr(quant_path, 'with_name'):
        sd_path = quant_path.with_name(quant_path.name + ".state_dict.pth")
    else:  # fallback if a string was passed
        sd_path = str(quant_path) + ".state_dict.pth"
    try:
        torch.save(q_model.state_dict(), sd_path)
        logger.info("Saved quantized state_dict backup -> %s", sd_path)
    except Exception as e:
        logger.warning("Failed to save quantized state_dict backup: %s", e)

    try:
        orig_size = os.path.getsize(orig_model_path)
    except Exception:
        orig_size = None
    try:
        q_size = os.path.getsize(quant_path)
    except Exception:
        q_size = None

    logger.info("Quantized model saved. sizes -> quant: %s bytes, orig: %s bytes", str(q_size), str(orig_size))
    return quant_path

def export_onnx(orig_model_path=BEST_MODEL_PATH, onnx_path=ONNX_MODEL_PATH, opset_version=13, device="cpu"):
    logger.info("Export ONNX: Loading float model (from %s)...", orig_model_path)
    if not os.path.exists(orig_model_path):
        raise FileNotFoundError(orig_model_path)
    feature_order, _, _, _ = load_artifacts()
    num_features = len(feature_order)

    model = DeepHybridModel(num_features=num_features)
    loaded = False
    try:
        sd = robust_torch_load(orig_model_path, map_location=device)
        if isinstance(sd, dict):
            try:
                model.load_state_dict(sd)
                loaded = True
                logger.info("Loaded state_dict into model for ONNX export.")
            except Exception:
                for k in ("state_dict", "model_state_dict", "weights"):
                    if k in sd and isinstance(sd[k], dict):
                        model.load_state_dict(sd[k])
                        loaded = True
                        logger.info("Loaded nested '%s' state_dict for ONNX export.", k)
                        break
    except Exception:
        pass

    if not loaded:
        try:
            obj = robust_torch_load(orig_model_path, map_location=device)
            if isinstance(obj, nn.Module):
                model = obj
                loaded = True
                logger.info("Loaded full module object for ONNX export.")
        except Exception:
            pass

    if not loaded:
        raise RuntimeError("Could not load float model for ONNX export.")

    model.eval()
    dummy = torch.randn(1, num_features, dtype=torch.float32, device=device)
    logger.info("Exporting ONNX to %s (opset=%d) ...", onnx_path, opset_version)
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["logits"],
            do_constant_folding=True,
            dynamic_axes={"input": {0: "batch", 1: "features"}, "logits": {0: "batch"}},
        )
        logger.info("ONNX export complete.")
    except Exception as e:
        logger.exception("ONNX export failed: %s", e)
        raise

    if onnx is not None:
        try:
            m = onnx.load(onnx_path)
            onnx.checker.check_model(m)
            logger.info("ONNX model check passed.")
        except Exception as e:
            logger.warning("ONNX checker failed: %s", e)

    # If Phase 2 parity JSON exists, surface its status here for operator visibility
    try:
        parity_path = PHASE2_DIR / 'onnx_parity.json'
        if parity_path.exists():
            with open(parity_path, 'r') as f:
                parity = json.load(f)
            if parity.get('passed'):
                logger.info("Existing ONNX parity summary: PASSED (n=%s max=%.2e mean=%.2e)",
                            parity.get('n'), parity.get('max_diff'), parity.get('mean_diff'))
            else:
                logger.warning("Existing ONNX parity summary indicates failure: %s", parity)
    except Exception as e:
        logger.debug("Could not read parity JSON: %s", e)
    return onnx_path

# ----------------------
# Robust load quantized model
# ----------------------
def load_quantized_model(model_path, device="cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    logger.info("Loading quantized model from %s", model_path)
    obj = None
    try:
        obj = robust_torch_load(model_path, map_location=device)
    except Exception as e:
        logger.warning("robust_torch_load failed: %s. Attempting fallback state_dict load.", e)
        if hasattr(model_path, 'with_name'):
            sd_path = model_path.with_name(model_path.name + ".state_dict.pth")
        else:
            sd_path = str(model_path) + ".state_dict.pth"
        if os.path.exists(sd_path):
            try:
                obj = robust_torch_load(sd_path, map_location=device)
            except Exception as e2:
                logger.exception("Failed to load fallback state_dict: %s", e2)
                raise
        else:
            raise

    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        feature_order, _, _, _ = load_artifacts()
        model = DeepHybridModel(num_features=len(feature_order))
        try:
            model.load_state_dict(obj)
            logger.info("Loaded state_dict into model.")
        except Exception as e:
            # Try common nested keys first
            for k in ("state_dict", "model_state_dict", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    try:
                        model.load_state_dict(obj[k])
                        logger.info("Loaded nested '%s' into model.", k)
                        break
                    except Exception:
                        pass
            else:
                # Detect a quantized state_dict (dynamic quantization) and attempt to load into a quantized model
                try:
                    if any((isinstance(kk, str) and (kk.endswith(".scale") or kk.endswith(".zero_point") or "._packed_params" in kk)) for kk in obj.keys()):
                        logger.info("Detected quantized state_dict; constructing quantized model to load it...")
                        base = DeepHybridModel(num_features=len(feature_order))
                        q_model = quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
                        try:
                            q_model.load_state_dict(obj)
                        except Exception:
                            # Try non-strict as last resort
                            missing_unexpected = q_model.load_state_dict(obj, strict=False)
                            logger.warning("Loaded quantized state_dict with non-strict matching (missing=%s unexpected=%s)", getattr(missing_unexpected, 'missing_keys', '?'), getattr(missing_unexpected, 'unexpected_keys', '?'))
                        model = q_model
                        logger.info("Loaded quantized state_dict into dynamically quantized model.")
                    else:
                        raise RuntimeError("State dict does not match float model and does not look quantized.")
                except Exception as qe:
                    logger.exception("Failed loading dict into model (quantized path as well): %s", qe)
                    raise RuntimeError("Quantized file could not be interpreted as nn.Module or state_dict.")
    else:
        raise RuntimeError("Loaded object is not a module or a dict.")

    model.to(device)
    model.eval()
    return model

# ----------------------
# ONNX Benchmark helper
# ----------------------
def benchmark_onnx(onnx_path, X_np, num_runs=100, cpus=1):
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Install onnxruntime to benchmark ONNX.")
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # warmups
    for _ in range(5):
        sess.run(None, {input_name: X_np[:1].astype(np.float32)})
    times = []
    for i in range(min(num_runs, X_np.shape[0])):
        s = time.perf_counter()
        _ = sess.run(None, {input_name: X_np[i:i+1].astype(np.float32)})
        e = time.perf_counter()
        times.append(e - s)
    times = np.array(times)
    return {
        "samples": int(times.size),
        "mean_ms": float(times.mean() * 1000),
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
    }

# ----------------------
# Benchmarking (simulate edge)
# ----------------------
def set_cpu_limit(cpus=1):
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpus)
    logger.info("Set CPU thread limits to %d via environment variables.", cpus)

def benchmark_model(model_path, split="val", num_samples=2000, cpus=1, device="cpu", use_onnx=False):
    set_cpu_limit(cpus)

    if not os.path.exists(PHASE1_DATA_PKL):
        raise FileNotFoundError(PHASE1_DATA_PKL)
    with open(PHASE1_DATA_PKL, "rb") as f:
        data = pickle.load(f)

    key_X = "X_" + split
    key_y = "y_" + split
    X = data.get(key_X, None)
    y = data.get(key_y, None)
    if X is None:
        X = data.get("X_val") if split == "val" else data.get("X_test") if split == "test" else data.get("X_train")
        y = data.get("y_val") if split == "val" else data.get("y_test") if split == "test" else data.get("y_train")
    if X is None:
        raise ValueError("Split not found in data.pkl")

    n = min(num_samples, X.shape[0])
    idx = np.random.choice(X.shape[0], n, replace=False)
    X_sample = X[idx]
    y_sample = y[idx] if y is not None else None

    feature_order, clip_low, clip_high, scaler = load_artifacts()
    
    # Load kept feature indices for pruned model compatibility
    eval_path = os.path.join(PHASE2_DIR, "evaluation.json")
    kept = list(range(X.shape[1]))
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                eval_json = json.load(f)
            kept = eval_json.get("meta", {}).get("kept_feature_indices", kept)
            kept = [int(i) for i in kept]
            logger.info("Benchmark: using %d kept features", len(kept))
        except Exception as e:
            logger.warning("Failed to load kept indices from eval.json: %s", e)
    
    X_sample = X_sample[:, np.array(kept)]
    # y_sample unchanged
    
    model_path_str = str(model_path)
    results = {
        "model_path": model_path_str,
        "split": split,
        "num_samples": int(n),
        "cpus": cpus,
        "device": device,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # env metadata
    try:
        torch_ver = torch.__version__
    except Exception:
        torch_ver = None
    try:
        ort_ver = ort.get_version() if ort is not None else None
    except Exception:
        ort_ver = None

    results["env"] = {
        "python": sys.version.split()[0],
        "platform": _platform.platform(),
        "torch": torch_ver,
        "onnxruntime": ort_ver
    }

    if use_onnx or model_path_str.lower().endswith(".onnx"):
        if not os.path.exists(model_path_str):
            raise FileNotFoundError(model_path_str)
        logger.info("Benchmarking ONNX model using onnxruntime...")
        onnx_res = benchmark_onnx(model_path_str, X_sample, num_runs=n, cpus=cpus)
        results.update({"onnx": onnx_res})
        if psutil:
            proc = psutil.Process()
            results["mem_mb"] = proc.memory_info().rss / (1024*1024)
    else:
        try:
            model = load_quantized_model(model_path_str, device=device)
            model.eval()
            # warmup
            with torch.no_grad():
                t0 = torch.tensor(X_sample[0:1], dtype=torch.float32).to(device)
                model(t0)
            times = []
            if psutil:
                mem_before = psutil.Process().memory_info().rss / (1024*1024)
            else:
                mem_before = None
            with torch.no_grad():
                for row in X_sample:
                    t = torch.tensor(row.reshape(1, -1), dtype=torch.float32).to(device)
                    s = time.perf_counter()
                    _ = model(t)
                    e = time.perf_counter()
                    times.append(e - s)
            if psutil:
                mem_after = psutil.Process().memory_info().rss / (1024*1024)
            else:
                mem_after = None
            times = np.array(times)
            results["pt_quant"] = {
                "samples": int(times.size),
                "mean_ms": float(times.mean() * 1000),
                "p50_ms": float(np.percentile(times, 50) * 1000),
                "p95_ms": float(np.percentile(times, 95) * 1000),
                "mem_before_mb": mem_before,
                "mem_after_mb": mem_after,
            }
        except Exception as e:
            logger.error("Quantized model load/inference failed: %s. Attempting dynamic quantization fallback...", e)
            # Try to quantize the best float model and benchmark that
            try:
                qp = quantize_and_save(orig_model_path=BEST_MODEL_PATH)
                if qp and os.path.exists(qp):
                    model = load_quantized_model(str(qp), device=device)
                    model.eval()
                    with torch.no_grad():
                        t0 = torch.tensor(X_sample[0:1], dtype=torch.float32).to(device)
                        model(t0)
                    times = []
                    if psutil:
                        mem_before = psutil.Process().memory_info().rss / (1024*1024)
                    else:
                        mem_before = None
                    with torch.no_grad():
                        for row in X_sample:
                            t = torch.tensor(row.reshape(1, -1), dtype=torch.float32).to(device)
                            s = time.perf_counter()
                            _ = model(t)
                            e = time.perf_counter()
                            times.append(e - s)
                    mem_after = psutil.Process().memory_info().rss / (1024*1024) if psutil else None
                    times = np.array(times)
                    results["pt_quant"] = {
                        "samples": int(times.size),
                        "mean_ms": float(times.mean() * 1000),
                        "p50_ms": float(np.percentile(times, 50) * 1000),
                        "p95_ms": float(np.percentile(times, 95) * 1000),
                        "mem_before_mb": mem_before,
                        "mem_after_mb": mem_after,
                        "note": "dynamic quantization fallback from BEST_MODEL_PATH"
                    }
                else:
                    raise RuntimeError("Dynamic quantization did not produce a model path")
            except Exception as e2:
                logger.error("Dynamic quantization fallback failed: %s. Attempting ONNX fallback...", e2)
                # Final fallback: ONNX benchmark if available
                try:
                    onnx_path = str(ONNX_MODEL_PATH)
                    if not os.path.exists(onnx_path):
                        onnx_path = str(export_onnx())
                    onnx_res = benchmark_onnx(onnx_path, X_sample, num_runs=n, cpus=cpus)
                    results.update({"onnx": onnx_res, "fallback": "onnx"})
                except Exception as e3:
                    logger.error("ONNX fallback failed: %s", e3)
                    results.update({"error": str(e), "fallback_error": str(e2), "onnx_error": str(e3)})

    try:
        with open(BENCHMARK_OUTPUT, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved benchmark summary to %s", BENCHMARK_OUTPUT)
    except Exception as e:
        logger.warning("Failed saving benchmark JSON: %s", e)
    try:
        logger.info("Benchmark finished: %s", json.dumps(results, indent=2, default=str))
    except Exception:
        logger.info("Benchmark finished (JSON serialization fallback): %s", results)
    return results

# ----------------------
# MQTT subscriber for inference (supports PyTorch .pth or ONNX .onnx)
# ----------------------
class MQTTInference:
    def __init__(self, model_path, mqtt_broker="localhost", mqtt_port=1883,
                 mqtt_topic="iot/traffic", predictions_topic=None,
                 log_csv=DEFAULT_LOG_CSV, cpulimit=1, device="cpu",
                 quantize_on_load=False,
                 health_topic=None,
                 health_interval=HEALTH_INTERVAL_DEFAULT,
                 mqtt_username=None,
                 mqtt_password=None,
                 mqtt_tls=False,
                 mqtt_ca_cert=None,
                 mqtt_client_cert=None,
                 mqtt_client_key=None,
                 mqtt_tls_insecure=False,
                 connect_retries=0,
                 connect_backoff_initial=1.0,
                 connect_backoff_max=30.0,
                 connect_backoff_mult=2.0,
                 connect_jitter=0.25,
                 alert_log_max_per_sec=2,
                 require_hash_match=False,
                 enable_metrics=True,
                 metrics_port=9208,
                 force_scale=True,
                 no_scale=False,
                 debug_features=0,
                 feature_order_override=None,
                 scaler_override=None,
                 evaluation_json_override=None,
                 enable_ensemble=False,
                 min_feature_presence=None):
        if mqtt is None:
            raise RuntimeError("paho-mqtt not installed; install via pip install paho-mqtt")

        # Artifacts & environment
        self.feature_order, self.clip_low, self.clip_high, self.scaler = load_artifacts(feature_order_override, scaler_override)
        set_cpu_limit(cpulimit)
        self.device = device
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic

        # Topics & health
        self.predictions_topic = predictions_topic or MQTT_PREDICTIONS_TOPIC_DEFAULT
        self.health_topic = health_topic or MQTT_HEALTH_TOPIC_DEFAULT
        self.health_interval = max(3, int(health_interval))
        self._last_health = 0.0
        self._start_time = time.time()
        self.cpu_limit = cpulimit

        # Security / auth
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
    # TLS removed (rollback)
        # Connection retry/backoff settings
        self._connect_retries = int(max(0, connect_retries))  # 0 = unlimited retries
        self._connect_backoff_initial = max(0.1, float(connect_backoff_initial))
        self._connect_backoff_max = max(self._connect_backoff_initial, float(connect_backoff_max))
        self._connect_backoff_mult = max(1.0, float(connect_backoff_mult))
        self._connect_jitter = min(max(0.0, float(connect_jitter)), 1.0)

        # Logging / quantization flags
        self.log_csv = log_csv
        self.quantize_on_load = quantize_on_load
        self._ensure_log_header()
        # Ensemble / thresholds
        self.enable_ensemble = bool(enable_ensemble)
        self._lgbm_model = None
        self.threshold_deep_only = None
        # Unified presence threshold
        self._min_feature_presence = float(min_feature_presence) if min_feature_presence is not None else float(os.getenv("MIN_FEATURE_PRESENCE", "0.9"))
        # Alert rate limiting tokens
        self.alert_log_max_per_sec = max(1, int(alert_log_max_per_sec))
        self._alert_tokens = self.alert_log_max_per_sec
        self._last_token_refill = time.time()
        self._last_inference_ms = None

        # Optional hash verification
        try:
            if require_hash_match:
                self._verify_hashes_strict(model_path)
            else:
                self._verify_hashes_best_effort(model_path)
        except Exception as e:
            logger.error("Hash verification failed: %s", e)
            if require_hash_match:
                raise

        # New: verify phase2 artifact manifest if present
        try:
            manifest_path = os.path.join(PHASE2_DIR, 'artifact_manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    man = json.load(f)
                mismatches = []
                for fname, h in man.items():
                    fp = os.path.join(PHASE2_DIR, fname)
                    if os.path.exists(fp):
                        ah = self._file_sha256(fp)
                        if ah != h:
                            mismatches.append((fname, h[:10], ah[:10]))
                if mismatches:
                    logger.warning("Artifact manifest mismatches detected: %s", mismatches)
                else:
                    logger.info("Artifact manifest integrity OK (%d files)", len(man))
        except Exception as e:
            logger.warning("Manifest verification failed: %s", e)

        # Optional signature enforcement
        if _phase_verify_or_exit is not None:
            try:
                _phase_verify_or_exit()
            except SystemExit:
                raise
            except Exception as _sig_e:
                logger.warning("Artifact signature verification skipped/failed: %s", _sig_e)

        # Load calibration from eval.json
        eval_path = evaluation_json_override if evaluation_json_override else os.path.join(PHASE2_DIR, "evaluation.json")
        self.temperature = 1.0
        self.threshold = 0.5
        self.already_standardized = False
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    eval_json = json.load(f)
                self.temperature = float(eval_json.get("calibration", {}).get("temperature", 1.0))
                self.threshold = float(eval_json.get("meta", {}).get("val_best_threshold", 0.5))
                self.threshold_deep_only = float(eval_json.get("meta", {}).get("val_best_threshold_deep_only", self.threshold))
                self.already_standardized = eval_json.get("meta", {}).get("already_standardized", False)
                # Decide active threshold (use deep-only when ensemble disabled)
                self.active_threshold = self.threshold if self.enable_ensemble else (self.threshold_deep_only or self.threshold)
                logger.info("Loaded temperature=%.4f, threshold=%.4f already_standardized=%s from eval.json",
                            self.temperature, self.threshold, self.already_standardized)
                # Future-proof: clamp sub-unity temperature (which would amplify logits) and retain original
                if self.temperature < 1.0:
                    self.original_temperature = self.temperature
                    self.temperature = 1.0
                    logger.warning("Clamped temperature %.4f -> 1.0 (temperatures <1 amplify logits, likely mis-saved).", self.original_temperature)
            except Exception as e:
                logger.warning("Failed to load eval.json: %s", e)

        model_path_str = str(model_path)
        # Special-case: quantized file names
        if "quantized" in model_path_str.lower():
            # keep temp = 1.0 for quantized (often needed) but still allow user override
            self.temperature = 1.0
            logger.info("Overriding temperature to 1.0 for quantized model (by filename heuristic)")

        self.is_onnx = model_path_str.lower().endswith(".onnx")
        REQUIRE_ONNX = os.getenv("REQUIRE_ONNX", "0").lower() in ("1","true","yes")
        _is_pi = _platform.machine().lower() in ("armv7l","aarch64","arm64")
        if self.is_onnx:
            if ort is None:
                msg = f"onnxruntime not available (import error: {_onnx_import_error})" if '_onnx_import_error' in globals() and _onnx_import_error else "onnxruntime not installed"
                if REQUIRE_ONNX:
                    raise RuntimeError(msg + " and REQUIRE_ONNX=1")
                logger.warning("%s – falling back to PyTorch model %s", msg, model_path_str)
                if _is_pi:
                    logger.warning("Raspberry Pi detected. Install aarch64/arm wheel: pip install onnxruntime==<version>")
                self.is_onnx = False  # force PyTorch branch
            else:
                try:
                    self.ort_sess = ort.InferenceSession(model_path_str, providers=["CPUExecutionProvider"])
                    self.ort_input = self.ort_sess.get_inputs()[0].name
                    logger.info("Loaded ONNX model for inference (onnxruntime).")
                    try:
                        rng = np.random.default_rng(0)
                        test_batch = rng.normal(size=(8, len(self.feature_order))).astype(np.float32)
                        out_raw = self.ort_sess.run(None, {self.ort_input: test_batch})[0]
                        ex = np.exp(out_raw - out_raw.max(axis=1, keepdims=True))
                        probs = ex / ex.sum(axis=1, keepdims=True)
                        if np.allclose(probs[:,1].std(), 0.0, atol=1e-7):
                            logger.warning("ONNX warmup: attack prob std ~0; check preprocessing alignment.")
                    except Exception as e:
                        logger.debug("Warmup parity probe skipped: %s", e)
                except Exception as e:
                    if REQUIRE_ONNX:
                        raise RuntimeError(f"Failed to init onnxruntime session: {e}")
                    logger.warning("Failed to init ONNX session (%s) – falling back to PyTorch.", e)
                    self.is_onnx = False
        else:
            if self.quantize_on_load:
                self.model = self._load_float_model(model_path, device)
                self.model.eval()
                logger.info("Applying dynamic quantization on-the-fly (skipping LSTM)...")
                self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
                self.model.to(device)
                self.model.eval()
                logger.info("On-the-fly quantized PyTorch model ready for inference on device=%s.", device)
            else:
                self.model = load_quantized_model(model_path, device=device)
                logger.info("Loaded PyTorch model for inference on device=%s.", device)

        # Attempt ensemble LightGBM load if enabled
        if self.enable_ensemble:
            try:
                import joblib
                # Search common candidate filenames
                candidates = [
                    os.path.join(PHASE2_DIR, 'lgbm_model.pkl'),
                    os.path.join(PHASE2_DIR, 'lgbm_model.joblib'),
                    os.path.join(PHASE2_DIR, 'lightgbm_model.pkl'),
                ]
                found = None
                for c in candidates:
                    if os.path.exists(c):
                        found = c
                        break
                if found:
                    self._lgbm_model = joblib.load(found)
                    logger.info("Loaded LightGBM ensemble model: %s", found)
                else:
                    logger.warning("Ensemble enabled but no LightGBM model file found among %s", [os.path.basename(x) for x in candidates])
            except Exception as e:
                logger.warning("Failed loading ensemble model: %s", e)
        # Ensure active threshold attr present
        if not hasattr(self, 'active_threshold'):
            self.active_threshold = self.threshold

        # New feature flag defaults; CLI may overwrite after object creation
        self._lock_noscale_if_standardized = False
        self._locked_noscale = False
        self._enable_temp_decay = False
        self._temp_decay_history = []  # store recent p99 gap observations (post-auto-temp)
        self._mode_logit_gaps = {}
        # Latency tracking & dynamic baseline flags
        self._latencies_ms = []
        self._dynamic_baseline_enabled = True
        self._baseline_recalibrated = False

        # MQTT client
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        if self.mqtt_username:
            try:
                self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
            except Exception:
                logger.warning("Failed to set MQTT username/password")
        # TLS configuration
        self._mqtt_tls = bool(mqtt_tls)
        self._mqtt_ca_cert = mqtt_ca_cert
        self._mqtt_client_cert = mqtt_client_cert
        self._mqtt_client_key = mqtt_client_key
        self._mqtt_tls_insecure = bool(mqtt_tls_insecure)
        if self._mqtt_tls:
            try:
                tls_kwargs = {}
                if self._mqtt_ca_cert:
                    tls_kwargs['ca_certs'] = self._mqtt_ca_cert
                if self._mqtt_client_cert and self._mqtt_client_key:
                    tls_kwargs['certfile'] = self._mqtt_client_cert
                    tls_kwargs['keyfile'] = self._mqtt_client_key
                # Configure TLS (ssl imported lazily)
                import ssl
                tls_kwargs['tls_version'] = ssl.PROTOCOL_TLS_CLIENT
                if self._mqtt_tls_insecure:
                    tls_kwargs['cert_reqs'] = ssl.CERT_NONE
                self.client.tls_set(**tls_kwargs)
                if self._mqtt_tls_insecure:
                    self.client.tls_insecure_set(True)
                logger.info("Configured MQTT TLS (insecure=%s ca=%s)", self._mqtt_tls_insecure, self._mqtt_ca_cert)
            except Exception as e:
                logger.error("Failed configuring TLS: %s", e)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # Counters & metrics
        self.total = 0
        self.correct = 0
        self._metrics_enabled = enable_metrics
        self._metrics_port = int(metrics_port)
        self._metrics_started = False
        self._metrics = {"messages_total": 0, "last_inference_ms": 0.0, "malformed_total": 0, "feature_presence_ratio_sum": 0.0}
        self._start_time_global = self._start_time
        self._inference_batches = 0
        # Logit gap statistics (for confidence distribution monitoring)
        self._logit_gaps = []  # store abs(logit1-logit0) before temperature scaling
        # Rolling accuracy guardrail buffers
        self._recent_preds = []
        self._recent_labels = []
        self._guardrail_window = 200
        self._baseline_acc = None

        # --- Contract validation & drift monitor ---
        self._contract = None
        self._drift_monitor = None
        # pipeline_contract module not found; skipping contract validation and drift monitor
        # --- End contract block ---

        # Graceful shutdown signals
        try:
            signal.signal(signal.SIGINT, self._graceful_exit)
            signal.signal(signal.SIGTERM, self._graceful_exit)
        except Exception:
            logger.debug("Signal handlers not set (likely not main thread)")
        # small counter for limiting verbose debug logs
        self._debug_counter = 0
        # Scaling / debug flags
        self.force_scale = bool(force_scale) and not bool(no_scale)
        self.no_scale = bool(no_scale)
        self.debug_features = int(debug_features) if debug_features else 0
        if self.no_scale:
            logger.info("Scaling mode: NO-SCALE (raw features passed to model)")
        else:
            logger.info("Scaling mode: %s", "FORCE" if self.force_scale else ("AUTO(already_standardized=%s)" % self.already_standardized))
        if self.debug_features > 0:
            logger.info("Will output feature debug for first %d messages", self.debug_features)
        # Domain detection buffers (raw vs standardized) - gather first N raw vectors pre-scaling
        self._domain_detection_done = False
        self._domain_samples = []  # list of np.ndarray raw feature vectors BEFORE scaling
        self._domain_detection_N = 25
        # Metrics server lazy start
        if self._metrics_enabled:
            self._start_metrics_server()
        # Alignment / adaptation buffers
        self._align_buffer = []  # list of raw feature vectors (np.ndarray shape (F,))
        self._align_labels = []  # corresponding int labels
        self._align_probe_done = False
        self._expected_val_acc = None
        try:
            # evaluation.json may have val_accuracy
            eval_json_local = locals().get('eval_json') or None
            if eval_json_local:
                self._expected_val_acc = float(eval_json_local.get('metrics', {}).get('val_accuracy', None) or eval_json_local.get('meta', {}).get('val_accuracy', None))
        except Exception:
            pass
        # Allow temperature override via env PHASE4_FORCE_TEMPERATURE
        try:
            forced_temp = os.getenv('PHASE4_FORCE_TEMPERATURE', None)
            if forced_temp is not None:
                self.temperature = float(forced_temp)
                logger.info("Overriding temperature from env -> %.4f", self.temperature)
        except Exception:
            pass

        # --- Guardrail auto-action state initialization ---
        self._guardrail_status = 0  # 0=ok 1=notice 2=alert
        self._guardrail_alert_streak = 0
        self._guardrail_last_action_ts = 0.0
        self._recalibration_flag = False
        self._scaling_mode_before_guardrail = None
        # --- Training logit gap reference for adaptive temperature ---
        self._train_logit_gap_p99 = None
        try:
            # evaluation.json may have been loaded into eval_json local earlier
            if 'eval_json' in locals():
                try:
                    self._train_logit_gap_p99 = float(eval_json.get('meta', {}).get('logit_gap', {}).get('p99', None))
                except Exception:
                    pass
            if self._train_logit_gap_p99 is None and self._contract is not None:
                try:
                    lg = getattr(self._contract, 'logit_gap', None)
                    if isinstance(lg, dict) and 'p99' in lg:
                        self._train_logit_gap_p99 = float(lg['p99'])
                except Exception:
                    pass
            if self._train_logit_gap_p99:
                logger.info("Training logit_gap p99 reference=%.3f", self._train_logit_gap_p99)
        except Exception:
            pass
        # Scaler preview stats (first 5)
        try:
            if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                logger.debug("Scaler mean[:5]=%s scale[:5]=%s", np.array(self.scaler.mean_)[:5].round(3).tolist(), np.array(self.scaler.scale_)[:5].round(3).tolist())
        except Exception:
            pass
        self._auto_calibrate_temperature = False

    def _start_metrics_server(self):
        if self._metrics_started:
            return
        handler_self = self
        class _P4MetricsHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                return
            def do_GET(self):
                if self.path == '/ready':
                    ready = (handler_self._inference_batches > 0) or (time.time()-handler_self._start_time_global > 5)
                    data = ("ready" if ready else "starting").encode(); code = 200 if ready else 503
                    self.send_response(code); self.send_header('Content-Type','text/plain'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data); return
                if self.path != '/metrics':
                    self.send_response(404); self.end_headers(); return
                m = handler_self._metrics
                lines = [
                    f"phase4_messages_total {m['messages_total']}",
                    f"phase4_last_inference_ms {m['last_inference_ms']}",
                    f"phase4_inference_batches_total {handler_self._inference_batches}",
                    f"phase4_uptime_seconds {int(time.time()-handler_self._start_time_global)}",
                    f"phase4_malformed_total {m.get('malformed_total',0)}",
                    f"phase4_feature_presence_ratio_avg { (m['feature_presence_ratio_sum']/m['messages_total']) if m['messages_total']>0 else 0.0}",
                    f"phase4_temperature_current {getattr(handler_self,'temperature',1.0)}",
                    f"phase4_mode_locked_noscale {1 if getattr(handler_self,'_locked_noscale', False) else 0}",
                ]
                # Latency percentiles
                try:
                    if handler_self._latencies_ms:
                        lats = np.array(handler_self._latencies_ms[-1000:])
                        lines.append(f"phase4_latency_ms_p50 {float(np.percentile(lats,50))}")
                        lines.append(f"phase4_latency_ms_p95 {float(np.percentile(lats,95))}")
                        lines.append(f"phase4_latency_ms_p99 {float(np.percentile(lats,99))}")
                except Exception:
                    pass
                # optional drift metrics
                try:
                    if handler_self._drift_monitor is not None:
                        ds = handler_self._drift_monitor.current_stats()
                        if 'mean_shift_avg' in ds:
                            lines.append(f"phase4_feature_mean_shift_avg {ds['mean_shift_avg']}")
                        if 'std_ratio_avg' in ds:
                            lines.append(f"phase4_feature_std_ratio_avg {ds['std_ratio_avg']}")
                        if 'psi_mean' in ds:
                            lines.append(f"phase4_feature_psi_mean {ds['psi_mean']}")
                        # Top-5 PSI features (if available)
                        try:
                            psi_last = getattr(handler_self._drift_monitor, '_psi_last', None)
                            if psi_last is not None and psi_last.size and isinstance(psi_last, np.ndarray):
                                feat_names = handler_self.feature_order
                                order = np.argsort(-np.nan_to_num(psi_last, nan=-1.0))[:5]
                                for rank, idx in enumerate(order, start=1):
                                    if idx < len(feat_names):
                                        lines.append(f"phase4_feature_psi_top{rank}{{feature=\"{feat_names[idx]}\"}} {psi_last[idx]}")
                        except Exception:
                            pass
                    # Logit gap stats (rolling)
                    try:
                        if handler_self._logit_gaps:
                            gaps_np = np.array(handler_self._logit_gaps[-500:])  # recent window
                            lines.append(f"phase4_logit_gap_mean {float(gaps_np.mean())}")
                            lines.append(f"phase4_logit_gap_p50 {float(np.percentile(gaps_np,50))}")
                            lines.append(f"phase4_logit_gap_p90 {float(np.percentile(gaps_np,90))}")
                            lines.append(f"phase4_logit_gap_p99 {float(np.percentile(gaps_np,99))}")
                    except Exception:
                        pass
                except Exception:
                    pass
                # Guardrail metrics
                try:
                    if handler_self._recent_preds and handler_self._recent_labels:
                        arr_p = np.array(handler_self._recent_preds[-handler_self._guardrail_window:])
                        arr_l = np.array(handler_self._recent_labels[-handler_self._guardrail_window:])
                        if arr_p.size == arr_l.size and arr_p.size > 0:
                            rw_acc = (arr_p == arr_l).mean()
                            lines.append(f"phase4_rolling_accuracy {rw_acc}")
                            if handler_self._baseline_acc:
                                lines.append(f"phase4_baseline_accuracy {handler_self._baseline_acc}")
                                lines.append(f"phase4_guardrail_window {handler_self._guardrail_window}")
                                lines.append(f"phase4_guardrail_status {handler_self._guardrail_status}")
                                lines.append(f"phase4_guardrail_streak {handler_self._guardrail_alert_streak}")
                                lines.append(f"phase4_recalibration_flag {1 if handler_self._recalibration_flag else 0}")
                except Exception:
                    pass
                data = ("\n".join(lines)+"\n").encode()
                self.send_response(200); self.send_header('Content-Type','text/plain; version=0.0.4'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        try:
            srv = socketserver.TCPServer(("0.0.0.0", self._metrics_port), _P4MetricsHandler)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            self._metrics_started = True
            logger.info("Phase 4 metrics exporter on :%d/metrics", self._metrics_port)
        except Exception as e:
            logger.warning("Phase 4 metrics server failed: %s", e)

    # -------- Hash verification helpers --------
    def _file_sha256(self, path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    def _verify_hashes_best_effort(self, model_path):
        manifest = os.path.join(PHASE1_DIR, 'manifest_hashes.json')
        if not os.path.exists(manifest):
            return
        try:
            with open(manifest, 'r') as f:
                man = json.load(f)
        except Exception:
            return
        targets = [
            ("model_hybrid.onnx", model_path if model_path.endswith('.onnx') else ONNX_MODEL_PATH),
            ("final_model_hybrid.pth", FINAL_MODEL_PATH),
            ("scaler.pkl", PHASE2_SCALER),
            ("feature_order.json", PHASE2_FEATURE_ORDER),
        ]
        for key, p in targets:
            p = str(p)
            if os.path.exists(p) and key in man:
                try:
                    actual = self._file_sha256(p)
                    if man[key] != actual:
                        logger.warning("Hash mismatch %s expected=%s actual=%s", key, man[key][:10], actual[:10])
                except Exception:
                    pass

    def _verify_hashes_strict(self, model_path):
        manifest = os.path.join(PHASE1_DIR, 'manifest_hashes.json')
        if not os.path.exists(manifest):
            raise RuntimeError("Manifest missing for strict hash verification")
        with open(manifest, 'r') as f:
            man = json.load(f)
        targets = [
            ("model_hybrid.onnx", model_path if model_path.endswith('.onnx') else ONNX_MODEL_PATH),
            ("final_model_hybrid.pth", FINAL_MODEL_PATH),
            ("scaler.pkl", PHASE2_SCALER),
            ("feature_order.json", PHASE2_FEATURE_ORDER),
        ]
        for key, p in targets:
            p = str(p)
            if os.path.exists(p) and key in man:
                actual = self._file_sha256(p)
                if man[key] != actual:
                    raise RuntimeError(f"Hash mismatch {key}")

    def _graceful_exit(self, *args):
        try:
            logger.info("Phase 4 shutting down...")
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        finally:
            try:
                sys.exit(0)
            except SystemExit:
                pass

    def _coerce_protocol_value(self, v):
        """Map protocol names to numeric codes aligned with Phase 1. OTHER=0, ICMP=1, TCP=6, UDP=17."""
        try:
            iv = int(v)
            return iv if iv in (1, 6, 17) else 0
        except Exception:
            s = str(v).strip().upper()
            # Try mapping artifact if present
            try:
                pm_path = os.path.join(PHASE2_DIR, "protocol_mapping.json")
                if os.path.exists(pm_path):
                    with open(pm_path, "r") as f:
                        pm = json.load(f)
                    if s in pm:
                        return int(pm[s])
            except Exception:
                pass
            return {"ICMP": 1, "TCP": 6, "UDP": 17}.get(s, 0)

    def preprocess_row(self, row_dict):
        # Handle nested features structure from MQTT publisher
        if 'features' in row_dict:
            features_dict = row_dict['features']
        else:
            features_dict = row_dict
        # Coerce values to numeric, especially 'protocol' possibly sent as name
        row_vals = []
        for k in self.feature_order:
            v = features_dict.get(k, 0.0)
            if k == "protocol":
                v = self._coerce_protocol_value(v)
            try:
                row_vals.append(float(v))
            except Exception:
                row_vals.append(0.0)
        x = np.array(row_vals, dtype=np.float32).reshape(1, -1)
        raw_x_initial = x.copy()  # preserve original before any clipping for domain detection
        raw_x = raw_x_initial.copy()
        # Sanitize NaN/Inf by imputing with midpoint of clip bounds
        try:
            if not np.all(np.isfinite(x)):
                mids = (self.clip_low + self.clip_high) / 2.0
                x = np.where(np.isfinite(x), x, mids)
        except Exception:
            pass
        # STRICT mode: bypass reconstruction & adaptive heuristics entirely
        if getattr(self, '_strict_full_scale', False):
            self._adaptive_mixed_scale = False
            self._auto_reconstruct_mixed = False
        # --- Domain detection sampling (pre-scaling) ---
        try:
            if not self._domain_detection_done:
                if len(self._domain_samples) < self._domain_detection_N:
                    self._domain_samples.append(raw_x[0].copy())
                if len(self._domain_samples) == self._domain_detection_N:
                    samples = np.vstack(self._domain_samples)  # shape (N, F)
                    # Heuristic: use port_src & port_dst ranges to infer domain.
                    # In RAW domain typical magnitudes >> 1000 frequently; in STANDARDIZED domain port values are ~[-1, +2].
                    def _feat_idx(name):
                        try:
                            return self.feature_order.index(name)
                        except ValueError:
                            return None
                    idx_ps = _feat_idx('port_src')
                    idx_pd = _feat_idx('port_dst')
                    raw_like_votes = 0
                    std_like_votes = 0
                    if idx_ps is not None:
                        ps_abs_med = float(np.median(np.abs(samples[:, idx_ps])))
                        # If median abs > 200 -> raw-like, else standardized-like
                        if ps_abs_med > 200:
                            raw_like_votes += 1
                        else:
                            std_like_votes += 1
                    if idx_pd is not None:
                        pd_abs_med = float(np.median(np.abs(samples[:, idx_pd])))
                        if pd_abs_med > 200:
                            raw_like_votes += 1
                        else:
                            std_like_votes += 1
                    # Additional heuristic: count features with |value| > 8 (unlikely after proper standardization)
                    large_val_ratio = float(np.mean(np.max(np.abs(samples), axis=1) > 8.0))
                    if large_val_ratio > 0.2:
                        raw_like_votes += 1
                    else:
                        std_like_votes += 1
                    domain = 'raw-like' if raw_like_votes >= std_like_votes else 'standardized-like'
                    rec = None
                    if domain == 'standardized-like' and self.force_scale and not self.no_scale:
                        rec = "Incoming features appear already standardized; consider --no-scale or --auto-scale to avoid double scaling."
                    elif domain == 'raw-like' and (self.no_scale or (not self.force_scale and not self.already_standardized)):
                        rec = "Incoming features appear raw-like; ensure scaling is enabled (remove --no-scale / use --force-scale)."
                    logger.warning("DOMAIN-DETECT result=%s votes(raw=%d std=%d) large_val_ratio=%.2f recommendation=%s", domain, raw_like_votes, std_like_votes, large_val_ratio, rec)
                    self._domain_detection_done = True
        except Exception:
            pass
        # After initial domain detection (may not yet be done for first N), decide clipping: skip clipping for standardized-like domain to avoid shrinking already standardized data
        inferred_domain = None
        try:
            if self._domain_detection_done and len(self._domain_samples) == self._domain_detection_N:
                samples = np.vstack(self._domain_samples)
                def _feat_idx(name):
                    try:
                        return self.feature_order.index(name)
                    except ValueError:
                        return None
                idx_ps = _feat_idx('port_src'); idx_pd = _feat_idx('port_dst')
                raw_votes = 0; std_votes = 0
                if idx_ps is not None:
                    med_ps = np.median(np.abs(samples[:, idx_ps]))
                    raw_votes += 1 if med_ps > 200 else 0
                    std_votes += 1 if med_ps <= 200 else 0
                if idx_pd is not None:
                    med_pd = np.median(np.abs(samples[:, idx_pd]))
                    raw_votes += 1 if med_pd > 200 else 0
                    std_votes += 1 if med_pd <= 200 else 0
                large_val_ratio = float(np.mean(np.max(np.abs(samples), axis=1) > 8.0))
                raw_votes += 1 if large_val_ratio > 0.2 else 0
                std_votes += 1 if large_val_ratio <= 0.2 else 0
                inferred_domain = 'raw-like' if raw_votes >= std_votes else 'standardized-like'
                # Optional auto-lock noscale to avoid double scaling
                if (inferred_domain == 'standardized-like' and self._lock_noscale_if_standardized \
                        and not self._locked_noscale and not getattr(self, '_strict_full_scale', False)):
                    self.no_scale = True
                    self.force_scale = False
                    self._locked_noscale = True
                    logger.warning("LOCK-NOSCALE: standardized-like domain confirmed (avoiding double scaling)")
        except Exception:
            inferred_domain = None
        try:
            if inferred_domain != 'standardized-like':
                x = np.clip(x, self.clip_low, self.clip_high)
            else:
                if self.debug_features and self._debug_counter < self.debug_features:
                    logger.debug("[CLIP-SKIP] standardized-like domain detected; skipping clipping")
        except Exception:
            pass
        raw_x = x.copy()
        # Decide scaling policy (simplified / Phase3-aligned)
        apply_scale = False
        # If user wants to ignore eval standardized flag treat as not standardized initially
        local_already_standardized = self.already_standardized and not getattr(self, '_ignore_eval_standardized', False)
        # If domain detector later flags standardized-like we will disable scaling automatically once.
        if not self.no_scale:
            if getattr(self, '_strict_full_scale', False):
                apply_scale = True
            elif self.force_scale:
                apply_scale = True
            else:
                mean_abs = float(np.mean(np.abs(x)))
                gstd = float(np.std(x))
                apply_scale = (mean_abs > 0.35 or gstd > 1.2) and not local_already_standardized
        # If domain detection concluded standardized-like and we are about to scale while eval says already_standardized -> skip to prevent double scaling
        try:
            if self._domain_detection_done:
                last_domain = None
                if len(self._domain_samples) == self._domain_detection_N:
                    # quick re-eval using last stored samples (reuse logic thresholds)
                    samples = np.vstack(self._domain_samples)
                    idx_ps = self.feature_order.index('port_src') if 'port_src' in self.feature_order else None
                    idx_pd = self.feature_order.index('port_dst') if 'port_dst' in self.feature_order else None
                    raw_votes = 0; std_votes = 0
                    if idx_ps is not None:
                        raw_votes += 1 if np.median(np.abs(samples[:,idx_ps])) > 200 else 0
                        std_votes += 1 if np.median(np.abs(samples[:,idx_ps])) <= 200 else 0
                    if idx_pd is not None:
                        raw_votes += 1 if np.median(np.abs(samples[:,idx_pd])) > 200 else 0
                        std_votes += 1 if np.median(np.abs(samples[:,idx_pd])) <= 200 else 0
                    large_val_ratio = float(np.mean(np.max(np.abs(samples), axis=1) > 8.0))
                    std_votes += 1 if large_val_ratio <= 0.2 else 0
                    raw_votes += 1 if large_val_ratio > 0.2 else 0
                    last_domain = 'raw-like' if raw_votes >= std_votes else 'standardized-like'
                if last_domain == 'standardized-like' and apply_scale and local_already_standardized:
                    if self.debug_features and self._debug_counter < self.debug_features:
                        logger.debug("[DOUBLE-SCALE-PREVENT] Skipping scaler.transform (domain=standardized-like, eval flag)")
                    apply_scale = False
                # Respect hard lock noscale
                if self._locked_noscale and apply_scale:
                    apply_scale = False
        except Exception:
            pass
        if apply_scale:
            try:
                x = self.scaler.transform(x)
            except Exception as e:
                logger.debug("Scaling failed: %s", e)
        scaled_x = x
        # Store debug snapshot for next on_message consumption
        if self.debug_features and self._debug_counter < self.debug_features:
            try:
                r_mean = float(raw_x.mean()); r_std = float(raw_x.std())
                s_mean = float(scaled_x.mean()); s_std = float(scaled_x.std())
                logger.debug("[FEATURE-DEBUG raw_mean=%.3f raw_std=%.3f scaled_mean=%.3f scaled_std=%.3f first_raw=%s first_scaled=%s]", r_mean, r_std, s_mean, s_std, raw_x[0,:5].round(3).tolist(), scaled_x[0,:5].round(3).tolist())
            except Exception:
                pass
        # retain raw for possible alignment probe
        self._last_raw_vector = raw_x[0].copy()
        return scaled_x

    # --- Alignment utilities ---
    def _predict_vector(self, raw_vec, mode="force"):
        """Return (prob_attack) using selected preprocessing mode on a single raw feature vector."""
        x = raw_vec.reshape(1, -1).astype(np.float32)
        temp = self.temperature
        if mode in ("force", "force_temp1") and not self.no_scale:
            try:
                x = self.scaler.transform(x)
            except Exception:
                pass
        if mode == "noscale":
            pass  # x stays raw
        if mode == "force_temp1":
            temp = 1.0
        # ONNX path only for now
        if self.is_onnx:
            out = self.ort_sess.run(None, {self.ort_input: x.astype(np.float32)})
            row = np.array(out[0])[0]
            # Treat as logits -> softmax
            logits = row / max(1e-6, float(temp))
            ex = np.exp(logits - np.max(logits))
            probs = ex / np.sum(ex)
            return float(probs[1])
        else:
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32).to(self.device)
                logits = self.model(t) / max(1e-6, float(temp))
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                return float(probs[1])

    def _run_alignment_probe(self):
        if self._align_probe_done:
            return
        if len(self._align_buffer) < 25:
            return
        if not any(l == 1 for l in self._align_labels) or not any(l == 0 for l in self._align_labels):
            return
        modes = ["force", "noscale", "force_temp1"]
        results = {}
        labels = np.array(self._align_labels)
        for m in modes:
            probs = [self._predict_vector(v, mode=m) for v in self._align_buffer]
            probs = np.array(probs)
            preds = (probs >= self.active_threshold).astype(int)
            acc = (preds == labels).mean() if np.all(np.isin(labels, [0,1])) else np.nan
            # separation metric: mean attack prob for label1 - label0
            try:
                sep = float(probs[labels==1].mean() - probs[labels==0].mean())
            except Exception:
                sep = float('nan')
            results[m] = {"accuracy": float(acc), "separation": sep}
        # Decide best mode by accuracy first then separation
        best_mode = max(results.items(), key=lambda kv: (np.nan_to_num(kv[1]['accuracy'], nan=-1), kv[1]['separation']))[0]
        current_mode = "noscale" if self.no_scale else ("force" if self.force_scale else "auto")
        logger.warning("ALIGN-PROBE results=%s current_mode=%s best_mode=%s", json.dumps(results), current_mode, best_mode)
        # Auto-switch if best differs and gains >=5% absolute accuracy
        try:
            cur_acc = results.get(current_mode, {}).get('accuracy', float('nan'))
            best_acc = results[best_mode]['accuracy']
            if not np.isnan(best_acc) and (np.isnan(cur_acc) or best_acc - cur_acc >= 0.05):
                if best_mode == "noscale":
                    self.no_scale = True; self.force_scale = False
                elif best_mode == "force_temp1":
                    self.no_scale = False; self.force_scale = True; self.temperature = 1.0
                else:  # force
                    self.no_scale = False; self.force_scale = True
                logger.warning("ALIGN-PROBE switching preprocessing to %s (temp=%.3f)", best_mode, self.temperature)
        except Exception:
            pass
        self._align_probe_done = True

    def _ensure_log_header(self):
        if not hasattr(self, '_rot_logger'):
            self._rot_logger = RotatingCSVLogger(Path(self.log_csv))
        if not os.path.exists(self.log_csv):  # header ensured by logger, but keep for safety
            try:
                with open(self.log_csv, "w", newline="") as f:
                    f.write("timestamp,true_label,pred,prob_attack\n")
            except Exception:
                pass

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("MQTT connected to %s:%d", self.mqtt_broker, self.mqtt_port)
            client.subscribe(self.mqtt_topic)
            logger.info("Subscribed to %s", self.mqtt_topic)
        else:
            logger.error("MQTT connection failed with rc=%s", rc)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            # ---------------- Input Validation (Priority1) ----------------
            features_container = payload.get('features') if isinstance(payload, dict) else None
            candidate = features_container if isinstance(features_container, dict) else payload
            present = [k for k in self.feature_order if k in candidate]
            presence_ratio = len(present) / max(1, len(self.feature_order))
            self._metrics['feature_presence_ratio_sum'] += presence_ratio
            if _metrics_mod is not None:
                try:
                    _metrics_mod.set_feature_presence(presence_ratio)
                except Exception:
                    pass
            if presence_ratio < self._min_feature_presence:
                self._metrics['malformed_total'] += 1
                if self.total % 50 == 0:
                    logger.warning(
                        "Malformed payload (feature_presence_ratio=%.2f < %.2f). Skipping.",
                        presence_ratio, self._min_feature_presence
                    )
                return
            # Simple numeric sanity clipping will happen inside preprocess_row (which already handles scaling/clipping)
            x_np = self.preprocess_row(payload)
            start_inf = time.time()
            if self.is_onnx:
                out = self.ort_sess.run(None, {self.ort_input: x_np.astype(np.float32)})
                raw = np.array(out[0])  # shape (1,2) expected
                row = raw[0]
                # Detect if raw already looks like probabilities (non-negative, sum ~1)
                row_sum = float(np.sum(row))
                if row.ndim == 1 and row.shape[0] == 2 and 0.999 <= row_sum <= 1.001 and np.all(row >= 0.0) and np.all(row <= 1.0):
                    # Treat as probabilities directly (avoid second softmax distortion)
                    probs = row
                else:
                    # Treat as logits; apply temperature scaling then softmax
                    # Record raw logit gap BEFORE temperature scaling
                    try:
                        self._logit_gaps.append(float(abs(row[1]-row[0])))
                        if len(self._logit_gaps) % 50 == 0:
                            gaps_np = np.array(self._logit_gaps)
                            logger.debug("[LOGIT-GAPS n=%d p50=%.3f p90=%.3f p99=%.3f max=%.3f]", len(gaps_np), np.percentile(gaps_np,50), np.percentile(gaps_np,90), np.percentile(gaps_np,99), gaps_np.max())
                    except Exception:
                        pass
                    logits = row / max(1e-6, float(self.temperature))
                    ex = np.exp(logits - np.max(logits))
                    probs = ex / np.sum(ex)
                prob_attack = float(probs[1])
                # Optional ensemble blend
                if self.enable_ensemble and self._lgbm_model is not None:
                    try:
                        lgbm_prob = float(self._lgbm_model.predict_proba(x_np)[0,1])
                        prob_attack = 0.5 * prob_attack + 0.5 * lgbm_prob
                    except Exception:
                        pass
                pred = int(prob_attack >= self.active_threshold)
                if self.total < 3:  # early debug samples
                    logger.debug("ONNX raw=%s probs=%s sum=%.4f detected_probs=%s", raw.tolist(), probs.tolist(), row_sum, 0.999 <= row_sum <= 1.001)
            else:
                with torch.no_grad():
                    t = torch.tensor(x_np, dtype=torch.float32).to(self.device)
                    logits = self.model(t) / max(1e-6, float(self.temperature))
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                    prob_attack = float(probs[1])
                    if self.enable_ensemble and self._lgbm_model is not None:
                        try:
                            lgbm_prob = float(self._lgbm_model.predict_proba(x_np)[0,1])
                            prob_attack = 0.5 * prob_attack + 0.5 * lgbm_prob
                        except Exception:
                            pass
                    pred = int(prob_attack >= self.active_threshold)
            self._last_inference_ms = (time.time() - start_inf) * 1000.0
            try:
                self._metrics['messages_total'] += 1
                self._metrics['last_inference_ms'] = float(self._last_inference_ms)
                self._inference_batches += 1
            except Exception:
                pass
            # Adaptive & robust temperature calibration
            try:
                if self._auto_calibrate_temperature and self._train_logit_gap_p99 and len(self._logit_gaps) >= 50 and (len(self._logit_gaps) % 50 == 0):
                    recent_gaps = np.array(self._logit_gaps[-500:])
                    raw_p99 = float(np.percentile(recent_gaps, 99))
                    clipped = np.clip(recent_gaps, None, 3.0 * self._train_logit_gap_p99)
                    robust_p99 = float(np.percentile(clipped, 99))
                    eff_p99 = min(raw_p99, robust_p99 * 1.05)
                    mode_label = 'noscale' if self.no_scale else ('force' if self.force_scale else 'auto')
                    self._mode_logit_gaps.setdefault(mode_label, []).append(raw_p99)
                    logger.debug("[LOGIT-GAPS n=%d p50=%.3f p90=%.3f p99=%.3f robust_p99=%.3f eff_p99=%.3f max=%.3f MODE=%s]", len(recent_gaps), np.percentile(recent_gaps,50), np.percentile(recent_gaps,90), raw_p99, robust_p99, eff_p99, recent_gaps.max(), mode_label)
                    is_quant = (not self.is_onnx) and ('quantized' in (getattr(self, 'model_path', '') or ''))
                    high_factor = 2.0 if is_quant else 1.5
                    temp_cap = 2.2 if is_quant else 3.0
                    if eff_p99 > high_factor * self._train_logit_gap_p99 and self.temperature < temp_cap:
                        target = eff_p99 / self._train_logit_gap_p99
                        new_temp = min(temp_cap, max(self.temperature + 0.15, min(temp_cap, target)))
                        if new_temp - self.temperature >= 0.05:
                            logger.warning("AUTO-TEMP increase %.3f -> %.3f (eff_p99=%.3f train_p99=%.3f quant=%s)", self.temperature, new_temp, eff_p99, self._train_logit_gap_p99, is_quant)
                            self.temperature = new_temp
                    elif eff_p99 < 0.8 * self._train_logit_gap_p99 and self.temperature > 1.05:
                        new_temp = max(1.0, self.temperature * 0.9)
                        if self.temperature - new_temp >= 0.05:
                            logger.info("AUTO-TEMP decrease %.3f -> %.3f (eff_p99=%.3f train_p99=%.3f)", self.temperature, new_temp, eff_p99, self._train_logit_gap_p99)
                            self.temperature = new_temp
                    if self._enable_temp_decay:
                        self._temp_decay_history.append(eff_p99)
                        if len(self._temp_decay_history) > 5:
                            self._temp_decay_history = self._temp_decay_history[-5:]
                        if len(self._temp_decay_history) >= 5 and all(p < 1.25 * self._train_logit_gap_p99 for p in self._temp_decay_history) and self.temperature > 1.4:
                            new_temp = max(1.3, round(self.temperature * 0.95, 3))
                            if self.temperature - new_temp >= 0.05:
                                logger.info("AUTO-TEMP decay %.3f -> %.3f (stable eff_p99 near training)", self.temperature, new_temp)
                                self.temperature = new_temp
                    if len(self._logit_gaps) % 100 == 0:
                        logger.info("CALIBRATION window=%d temp=%.3f eff_p99=%.3f robust_p99=%.3f raw_p99=%.3f", len(self._logit_gaps), self.temperature, eff_p99, robust_p99, raw_p99)
                        # Persist adaptive state snapshot (temperature, baseline, mode, counts)
                        try:
                            rs = globals().get('RUNTIME_STATE', {})
                            rs.update({
                                'updated_utc': datetime.utcnow().isoformat() + 'Z',
                                'temperature': float(self.temperature),
                                'baseline_acc': float(self._baseline_acc) if self._baseline_acc else None,
                                'mode_locked_noscale': bool(getattr(self, '_locked_noscale', False)),
                                'total_messages': int(getattr(self, 'total', 0)),
                                'logit_gap_count': len(self._logit_gaps),
                                'recent_eff_p99': float(eff_p99),
                                'recent_raw_p99': float(raw_p99),
                                'recent_robust_p99': float(robust_p99)
                            })
                            saver = globals().get('RUNTIME_STATE_SAVER')
                            if saver:
                                saver(rs)
                        except Exception as _e:
                            logger.debug("Runtime state save skipped: %s", _e)
            except Exception:
                pass
            ts = datetime.utcnow().isoformat()
            try:
                self._ensure_log_header()
                self._rot_logger.log(ts, payload.get('label', -1), pred, prob_attack)
            except Exception as _log_e:
                logger.debug("Prediction log write skipped: %s", _log_e)
            self.total += 1
            if self.debug_features and self._debug_counter < self.debug_features:
                # Log logits/probs snapshot
                try:
                    if 'row' in locals():
                        logger.debug("[LOGIT-DEBUG logits=%s probs=%s prob_attack=%.4f]", np.array(row).round(4).tolist(), np.array(probs).round(4).tolist(), prob_attack)
                except Exception:
                    pass
                self._debug_counter += 1
            # Drift monitor update
            try:
                if self._drift_monitor is not None and hasattr(self, '_last_raw_vector'):
                    self._drift_monitor.update(self._last_raw_vector)
                    if self.total % 100 == 0 and self.total > 0:
                        ds = self._drift_monitor.current_stats()
                        logger.debug("[DRIFT mean_shift_avg=%.4f std_ratio_avg=%.4f psi_mean=%s n=%d]", ds.get('mean_shift_avg', float('nan')), ds.get('std_ratio_avg', float('nan')), str(ds.get('psi_mean', 'nan')), ds.get('count', 0))
                        if _metrics_mod is not None:
                            try:
                                ms = ds.get('mean_shift_avg')
                                sr = ds.get('std_ratio_avg')
                                if isinstance(ms, (int,float)):
                                    _metrics_mod.set_drift_mean_shift(float(ms))
                                if isinstance(sr, (int,float)):
                                    _metrics_mod.set_drift_std_ratio(float(sr))
                            except Exception:
                                pass
                        # Threshold-based warnings (rate-limited by existing alert token bucket)
                        try:
                            psi_mean = ds.get('psi_mean', None)
                            mean_shift = ds.get('mean_shift_avg', None)
                            warn_msgs = []
                            if psi_mean is not None and isinstance(psi_mean, float) and psi_mean > 0.2:
                                warn_msgs.append(f"PSI drift high psi_mean={psi_mean:.3f} (>0.2)")
                            if mean_shift is not None and isinstance(mean_shift, float) and mean_shift > 0.5:
                                warn_msgs.append(f"Mean shift high mean_shift_avg={mean_shift:.3f} (>0.5)")
                            for mwarn in warn_msgs:
                                if self._alert_tokens > 0:  # reuse alert token bucket
                                    self._alert_tokens -= 1
                                    logger.warning("DRIFT-ALERT %s", mwarn)
                        except Exception:
                            pass
            except Exception:
                pass
            label_true = None
            try:
                if "label" in payload:
                    label_true = int(payload.get("label", -1))
            except Exception:
                label_true = None
            # Buffer for alignment probe
            try:
                if label_true is not None and label_true in (0,1) and hasattr(self, '_last_raw_vector'):
                    if len(self._align_buffer) < 120:  # cap buffer
                        self._align_buffer.append(self._last_raw_vector.copy())
                        self._align_labels.append(label_true)
            except Exception:
                pass
            if label_true is not None and label_true != -1 and pred == label_true:
                self.correct += 1
            acc = (self.correct / self.total * 100) if self.total > 0 else 0.0
            # Rolling window accuracy guardrail
            try:
                if label_true is not None and label_true in (0,1):
                    self._recent_preds.append(pred)
                    self._recent_labels.append(label_true)
                    if len(self._recent_preds) > self._guardrail_window:
                        self._recent_preds = self._recent_preds[-self._guardrail_window:]
                        self._recent_labels = self._recent_labels[-self._guardrail_window:]
                    if self._baseline_acc and len(self._recent_preds) >= max(50, self._guardrail_window//2):
                        arr_p = np.array(self._recent_preds)
                        arr_l = np.array(self._recent_labels)
                        rw_acc = (arr_p == arr_l).mean()
                        # After warmup (>=300 msgs) optionally recalibrate baseline once to a realistic target
                        if self._dynamic_baseline_enabled and not self._baseline_recalibrated and self.total >= 300:
                            new_base = max(0.94, min(0.99, rw_acc + 0.02))
                            if self._baseline_acc and abs(new_base - self._baseline_acc) > 0.01:
                                logger.info("BASELINE-ADAPT old=%.4f new=%.4f rw_acc=%.4f", self._baseline_acc, new_base, rw_acc)
                                self._baseline_acc = new_base
                                self._baseline_recalibrated = True
                        thresh90 = 0.90 * self._baseline_acc
                        thresh95 = 0.95 * self._baseline_acc
                        if rw_acc < thresh90 and self._alert_tokens > 0:
                            self._alert_tokens -= 1
                            logger.warning("GUARDRAIL-ALERT rolling_accuracy=%.4f < 90%% baseline=%.4f (window=%d)", rw_acc, self._baseline_acc, len(arr_p))
                            self._guardrail_status = 2
                            self._guardrail_alert_streak += 1
                        elif rw_acc < thresh95:
                            logger.info("Guardrail notice: rolling_accuracy=%.4f below 95%% baseline=%.4f (window=%d)", rw_acc, self._baseline_acc, len(arr_p))
                            self._guardrail_status = 1
                            self._guardrail_alert_streak = 0
                        else:
                            self._guardrail_status = 0
                            self._guardrail_alert_streak = 0
                        # Auto-actions on persistent alert
                        try:
                            now_ts = time.time()
                            if self._guardrail_status == 2:
                                # If sustained alert for >=3 consecutive alert evaluations and 10s since last action
                                if self._guardrail_alert_streak >= 3 and (now_ts - self._guardrail_last_action_ts) > 10:
                                    logger.warning("GUARDRAIL-ACTION triggered (streak=%d) executing remediation sequence", self._guardrail_alert_streak)
                                    # 1. Force alignment probe if not done
                                    if not self._align_probe_done:
                                        try:
                                            self._run_alignment_probe()
                                        except Exception:
                                            logger.debug("Alignment probe failed during guardrail action", exc_info=True)
                                    # 2. Adaptive scaling switch: if currently forcing scale and accuracy low, try temp=1 tweak; else if raw, try enabling scale
                                    try:
                                        if self.force_scale and self.temperature != 1.0:
                                            logger.warning("GUARDRAIL-ACTION: forcing temperature to 1.0 for robustness test")
                                            self.temperature = 1.0
                                        elif self.no_scale:
                                            logger.warning("GUARDRAIL-ACTION: enabling scaling (was no_scale mode)")
                                            self.no_scale = False; self.force_scale = True
                                        elif not self.force_scale:
                                            logger.warning("GUARDRAIL-ACTION: forcing scaling ON")
                                            self.force_scale = True
                                    except Exception:
                                        pass
                                    # 3. Mark recalibration flag
                                    self._recalibration_flag = True
                                    self._guardrail_last_action_ts = now_ts
                            else:
                                # Reset streak if not in alert (status 2)
                                if self._guardrail_status != 2:
                                    self._guardrail_alert_streak = 0
                        except Exception:
                            pass
            except Exception:
                pass
            # Trigger alignment probe if accuracy seems low relative to expected (if known)
            try:
                if (not self._align_probe_done) and self._expected_val_acc and self.total >= 30:
                    if acc < (self._expected_val_acc * 100 * 0.9):  # degrade threshold: 90% of expected
                        self._run_alignment_probe()
                elif (not self._align_probe_done) and self.total == 60:
                    # fallback unconditional probe if not yet run
                    self._run_alignment_probe()
            except Exception:
                pass
            # Alert rate limiting (token bucket)
            now_sec = time.time()
            if now_sec - self._last_token_refill >= 1.0:
                refill = int(now_sec - self._last_token_refill) * self.alert_log_max_per_sec
                self._alert_tokens = min(self.alert_log_max_per_sec, self._alert_tokens + refill)
                self._last_token_refill = now_sec
            if prob_attack > self.active_threshold and self._alert_tokens > 0:
                self._alert_tokens -= 1
                logger.warning("ALERT prob_attack=%.3f pred=%d ts=%s", prob_attack, pred, ts)
            logger.info("Msg#%d pred=%d prob_attack=%.4f acc=%.2f%% (true_label=%s)", self.total, pred, prob_attack, acc, str(label_true))
            # Publish prediction for Phase 5 dashboard
            try:
                out = {
                    "timestamp": ts,
                    "schema_version": "1.0",
                    "model_version": str(getattr(self, 'model_version', 'unknown')),
                    "pred": int(pred),
                    "prediction": int(pred),  # alias key for Phase 5 consumer expecting 'prediction'
                    "prob_attack": float(prob_attack),
                    "inference_ms": float(self._last_inference_ms) if self._last_inference_ms is not None else None,
                    "threshold": float(getattr(self, 'threshold', 0.5)),
                    "temperature": float(getattr(self, 'temperature', 1.0)),
                    "ensemble_enabled": bool(getattr(self, 'enable_ensemble', False)),
                }
                if label_true is not None and label_true != -1:
                    out["true_label"] = int(label_true)
                topic = getattr(self, 'predictions_topic', None)
                if not topic:
                    logger.warning("Prediction not published: predictions_topic unset (payload keys=%s)", list(out.keys()))
                else:
                    payload = json.dumps(out)
                    # Log first few publishes for diagnostics and then every 100th
                    if self.total < 5 or self.total % 100 == 0:
                        logger.info("Publishing prediction to %s: %s", topic, payload)
                    retain_flag = bool(getattr(self, '_retain_predictions', False))
                    result = client.publish(topic, payload, qos=0, retain=retain_flag)
                    # paho returns MQTTMessageInfo with .rc
                    try:
                        rc = getattr(result, 'rc', None)
                        if rc not in (0, None):
                            logger.warning("MQTT publish rc=%s topic=%s", rc, topic)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Predictions publish failed: %s", e)
            # Periodic health heartbeat
            now = time.time()
            if now - self._last_health >= self.health_interval:
                self._last_health = now
                try:
                    health_payload = {
                        "timestamp": ts,
                        "uptime_sec": int(now - self._start_time),
                        "model_type": "onnx" if self.is_onnx else "pytorch",
                        "messages": self.total,
                        "accuracy_pct": round(acc, 2),
                        "cpu_limit": self.cpu_limit,
                        "last_inference_ms": float(self._last_inference_ms) if self._last_inference_ms is not None else None,
                        "temperature": float(getattr(self, 'temperature', 1.0)),
                        "active_threshold": float(getattr(self, 'active_threshold', getattr(self, 'threshold', 0.5))),
                        "ensemble_enabled": bool(getattr(self, 'enable_ensemble', False)),
                        "min_feature_presence": float(getattr(self, '_min_feature_presence', 0.0)),
                        "malformed_total": int(self._metrics.get('malformed_total',0)),
                        "feature_presence_ratio_avg": (self._metrics['feature_presence_ratio_sum']/self._metrics['messages_total']) if self._metrics['messages_total']>0 else 0.0,
                        "mode_locked_noscale": bool(getattr(self, '_locked_noscale', False)),
                    }
                    # Latency percentiles snapshot
                    try:
                        if self._latencies_ms:
                            lats = np.array(self._latencies_ms[-500:])
                            health_payload["latency_ms_p50"] = float(np.percentile(lats,50))
                            health_payload["latency_ms_p95"] = float(np.percentile(lats,95))
                            health_payload["latency_ms_p99"] = float(np.percentile(lats,99))
                    except Exception:
                        pass
                    # Guardrail & drift enrichment
                    try:
                        if self._baseline_acc:
                            health_payload["baseline_acc"] = float(self._baseline_acc)
                        if self._recent_preds and self._recent_labels:
                            arr_p = np.array(self._recent_preds[-self._guardrail_window:])
                            arr_l = np.array(self._recent_labels[-self._guardrail_window:])
                            if arr_p.size == arr_l.size and arr_p.size > 0:
                                health_payload["rolling_accuracy"] = float((arr_p == arr_l).mean())
                        health_payload["guardrail_status"] = int(getattr(self, '_guardrail_status', 0))
                        health_payload["guardrail_streak"] = int(getattr(self, '_guardrail_alert_streak', 0))
                        health_payload["recalibration_flag"] = bool(getattr(self, '_recalibration_flag', False))
                        if self._drift_monitor is not None:
                            ds = self._drift_monitor.current_stats()
                            for k in ("mean_shift_avg","std_ratio_avg","psi_mean"):
                                if k in ds:
                                    health_payload[k] = float(ds[k]) if ds[k] is not None else None
                    except Exception:
                        pass
                    client.publish(self.health_topic, json.dumps(health_payload), qos=0)
                except Exception as e:
                    logger.debug("Health publish failed: %s", e)
        except Exception as e:
            logger.exception("Failed to handle incoming MQTT message: %s", e)

    def _softmax_np(self, logits):
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _load_float_model(self, path, device):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        num_features = len(self.feature_order)
        model = DeepHybridModel(num_features=num_features)
        loaded = None
        try:
            sd = robust_torch_load(path, map_location=device)
            if isinstance(sd, dict):
                try:
                    model.load_state_dict(sd)
                    loaded = "state_dict"
                    logger.info("Loaded state_dict into float model.")
                except Exception:
                    for k in ("state_dict", "model_state_dict", "weights"):
                        if k in sd and isinstance(sd[k], dict):
                            model.load_state_dict(sd[k])
                            loaded = k
                            logger.info("Loaded nested '%s' state_dict into float model.", k)
                            break
        except Exception as e:
            logger.debug("state_dict load attempt failed/raised: %s", e)

        if loaded is None:
            try:
                obj = robust_torch_load(path, map_location=device)
                if isinstance(obj, nn.Module):
                    model = obj
                    loaded = "module_object"
                    logger.info("Loaded full module object as float model from checkpoint.")
                else:
                    if isinstance(obj, dict):
                        raise RuntimeError("Loaded dict could not be applied to model.")
                    else:
                        raise RuntimeError("Checkpoint is of unexpected type: %s" % type(obj))
            except Exception as e:
                logger.exception("Failed to interpret checkpoint as state_dict or module: %s", e)
                raise

        model.to(device)
        model.eval()
        return model

    def start(self):
        attempt = 0
        backoff = self._connect_backoff_initial
        while True:
            attempt += 1
            try:
                self.client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
                if _metrics_mod and attempt > 1:
                    _metrics_mod.inc_reconnect()
                if attempt > 1:
                    logger.info("MQTT connected on attempt %d", attempt)
                logger.info("Starting MQTT loop forever...")
                self.client.loop_forever()
                break  # loop_forever exits only on error (then we retry)
            except Exception as e:
                logger.warning("MQTT connect attempt %d failed: %s", attempt, e)
                if self._connect_retries and attempt >= self._connect_retries:
                    logger.error("Exceeded max connection attempts (%d); aborting.", self._connect_retries)
                    raise
                sleep_for = backoff
                if self._connect_jitter > 0:
                    jitter_amt = sleep_for * self._connect_jitter
                    sleep_for = sleep_for + random.uniform(-jitter_amt, jitter_amt)
                    sleep_for = max(0.1, sleep_for)
                logger.info("Retrying MQTT connect in %.2fs (base=%.2fs attempts=%d limit=%s)", sleep_for, backoff, attempt, ("unlimited" if self._connect_retries == 0 else self._connect_retries))
                time.sleep(sleep_for)
                backoff = min(backoff * self._connect_backoff_mult, self._connect_backoff_max)
            # update static gauges each cycle
            if _metrics_mod:
                try:
                    _metrics_mod.set_threshold(float(self.active_threshold))
                    _metrics_mod.set_temperature(float(self.temperature))
                except Exception:
                    pass

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force-scale", action="store_true", help="(Default) Always apply scaler (legacy alignment)")
    p.add_argument("--quantize", action="store_true", help="Quantize FINAL_MODEL_PATH -> QUANT_MODEL_PATH")
    p.add_argument("--export-onnx", action="store_true", help="Export float PyTorch model to ONNX")
    p.add_argument("--benchmark", action="store_true", help="Benchmark quantized PyTorch model")
    p.add_argument("--onnx-benchmark", action="store_true", help="Benchmark an ONNX model using onnxruntime")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--cpulimit", type=int, default=1, help="Simulated CPU cores for inference (OMP threads)")
    p.add_argument("--run-mqtt", action="store_true", help="Run MQTT subscriber using model-path")
    # Allow legacy/short aliases for convenience (--model)
    p.add_argument("--model-path", "--model", dest="model_path", default=BEST_MODEL_PATH,
                   help="Path to .pth or .onnx model for MQTT (aliases: --model). Default is best non-quantized for accuracy")
    p.add_argument("--quantize-on-load", action="store_true", help="Load best float model and apply dynamic quantization on-the-fly for MQTT inference")
    # Provide short aliases to match user expectations (--broker, --port, --topic, --predictions-topic, --health-topic)
    p.add_argument("--mqtt-broker", "--broker", dest="mqtt_broker", default="localhost", help="MQTT broker hostname/IP (alias: --broker)")
    p.add_argument("--mqtt-port", "--port", dest="mqtt_port", type=int, default=1883, help="MQTT broker port (alias: --port)")
    p.add_argument("--mqtt-topic", "--topic", dest="mqtt_topic", default="iot/traffic", help="Ingress topic to subscribe (alias: --topic)")
    p.add_argument("--mqtt-predictions-topic", "--predictions-topic", dest="mqtt_predictions_topic", default=MQTT_PREDICTIONS_TOPIC_DEFAULT, help="Topic to publish predictions (alias: --predictions-topic)")
    p.add_argument("--mqtt-retain-predictions", action="store_true", help="Publish predictions with retain=True so late subscribers (dashboard) see last message")
    p.add_argument("--mqtt-health-topic", "--health-topic", dest="mqtt_health_topic", default=MQTT_HEALTH_TOPIC_DEFAULT, help="Topic for health heartbeats (alias: --health-topic)")
    p.add_argument("--health-interval", type=int, default=HEALTH_INTERVAL_DEFAULT, help="Seconds between health heartbeats")
    p.add_argument("--mqtt-username", default=None)
    p.add_argument("--mqtt-password", default=None)
    # TLS / Security (reintroduced w/ optional mutual auth)
    p.add_argument("--mqtt-tls", action="store_true", help="Enable TLS encryption for MQTT")
    p.add_argument("--mqtt-ca-cert", dest="mqtt_ca_cert", default=None, help="CA certificate path for verifying broker")
    p.add_argument("--mqtt-client-cert", dest="mqtt_client_cert", default=None, help="Client certificate for mutual TLS")
    p.add_argument("--mqtt-client-key", dest="mqtt_client_key", default=None, help="Client private key for mutual TLS")
    p.add_argument("--mqtt-tls-insecure", action="store_true", help="Skip TLS hostname verification (NOT for production)")
    p.add_argument("--require-auth", action="store_true", help="Fail start if username/password (and TLS when --mqtt-tls) not supplied")
    p.add_argument("--model-version", dest="model_version", default=None, help="Model version string embedded in publishes")
    # Connection robustness / retry flags
    p.add_argument("--connect-retries", type=int, default=0, help="Max MQTT connection attempts before giving up (0 = unlimited)")
    p.add_argument("--connect-backoff-initial", type=float, default=1.0, help="Initial backoff seconds for first retry")
    p.add_argument("--connect-backoff-max", type=float, default=30.0, help="Maximum backoff seconds")
    p.add_argument("--connect-backoff-mult", type=float, default=2.0, help="Backoff growth multiplier")
    p.add_argument("--connect-jitter", type=float, default=0.25, help="Jitter fraction (0-1) applied to each sleep interval")
    p.add_argument("--log", default=DEFAULT_LOG_CSV)
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    # Scaling / debug feature flags
    p.add_argument("--no-scale", action="store_true", help="Disable scaling (pass raw features)")
    p.add_argument("--auto-scale", action="store_true", help="Use heuristic auto scaling (overrides --force-scale)")
    p.add_argument("--debug-features", type=int, default=0, help="Log feature stats + logits for first N messages")
    p.add_argument("--adaptive-mixed-scale", action="store_true", help="Experimental: detect mixed-domain rows (some raw-like, some standardized) and scale only raw-like subset")
    p.add_argument("--auto-reconstruct-mixed", action="store_true", help="Attempt to reconstruct raw-like magnitudes from standardized-looking inputs before scaling (for pipelines where training expected raw->scale but stream already standardized once)")
    # New hard override flags to simplify troubleshooting
    p.add_argument("--strict-full-scale", action="store_true", help="Bypass all adaptive/reconstruction heuristics and always apply scaler exactly once (Phase3-style)")
    p.add_argument("--ignore-eval-standardized", action="store_true", help="Ignore eval.json already_standardized flag (treat incoming as raw unless domain detector forces standardized)")
    p.add_argument("--auto-calibrate-temperature", action="store_true", help="Adapt temperature based on observed logit gap p99 vs training p99 to mitigate overconfidence")
    p.add_argument("--lock-noscale-if-standardized", action="store_true", help="If domain detector concludes standardized-like, permanently disable scaling (avoid double scaling)")
    p.add_argument("--enable-temp-decay", action="store_true", help="Allow temperature to decay downward once logit gap p99 stabilizes near training distribution")
    # New explicit artifact override arguments for clarity
    p.add_argument("--onnx-path", dest="onnx_path", default=ONNX_MODEL_PATH, help="Override default ONNX model path for export/benchmark (defaults to artifacts_phase2/model_hybrid.onnx)")
    p.add_argument("--scaler", dest="scaler_path", default=None, help="Optional explicit scaler.pkl/json path override")
    p.add_argument("--feature-order", dest="feature_order_path", default=None, help="Optional explicit feature_order.json path override")
    p.add_argument("--evaluation-json", dest="evaluation_json_path", default=None, help="Optional explicit evaluation.json path override for threshold/temperature")
    p.add_argument("--enable-ensemble", action="store_true", help="Enable LightGBM ensemble blending if available")
    p.add_argument("--min-feature-presence", type=float, default=None, help="Minimum feature presence ratio required (default env MIN_FEATURE_PRESENCE or 0.9)")
    return p.parse_args()

def main():
    args = parse_args()

    # Elevate logging to DEBUG if feature debugging requested
    if getattr(args, 'debug_features', 0) and args.debug_features > 0:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled (debug_features=%d)", args.debug_features)

    # ---------------- Runtime state persistence (Priority1) ----------------
    # Persist adaptive calibration state so restarts reuse prior temperature, baseline accuracy, scaling mode decisions.
    RUNTIME_STATE_PATH = os.path.join(PHASE2_DIR, 'runtime_state.json')
    runtime_state = {}
    if os.path.exists(RUNTIME_STATE_PATH):
        try:
            with open(RUNTIME_STATE_PATH, 'r', encoding='utf-8') as fh:
                runtime_state = json.load(fh)
            logger.info("Loaded runtime_state.json keys=%s", list(runtime_state.keys()))
        except Exception as e:
            logger.warning("Failed to load runtime_state.json: %s", e)

    def _save_runtime_state(state: dict):
        tmp = RUNTIME_STATE_PATH + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(state, fh, indent=2)
            os.replace(tmp, RUNTIME_STATE_PATH)
        except Exception as e:
            logger.warning("Could not persist runtime_state: %s", e)

    # Expose for subscriber logic (will be referenced inside message handling)
    globals()['RUNTIME_STATE'] = runtime_state
    globals()['RUNTIME_STATE_SAVER'] = _save_runtime_state

    if args.quantize:
        try:
            qpath = quantize_and_save()
            logger.info("Quantization finished -> %s", qpath)
        except Exception as e:
            logger.exception("Quantization failed: %s", e)
            sys.exit(1)

    # Security enforcement checks (Priority1-MQTT security)
    if getattr(args, 'require_auth', False) and not (args.mqtt_username and args.mqtt_password):
        logger.error("--require-auth specified but username/password not provided. Aborting.")
        sys.exit(2)
    # If TLS requested but no CA cert, warn (allow for public CAs though)
    if getattr(args, 'mqtt_tls', False) and not args.mqtt_ca_cert:
        logger.warning("--mqtt-tls enabled without --mqtt-ca-cert; relying on system CA store")
    if getattr(args, 'mqtt_tls', False) and args.mqtt_tls_insecure:
        logger.warning("Insecure TLS verification disabled; NOT for production")
    if not (args.mqtt_username and args.mqtt_password):
        logger.warning("MQTT authentication is NOT set. Use --mqtt-username/--mqtt-password (and --require-auth) for production.")

    if args.export_onnx:
        try:
            onnx_path = export_onnx()
            logger.info("ONNX export finished -> %s", onnx_path)
        except Exception as e:
            logger.exception("ONNX export failed: %s", e)
            sys.exit(1)

    if args.benchmark:
        model_path = QUANT_MODEL_PATH
        if not os.path.exists(model_path):
            logger.info("Quantized model not found; quantizing from best model first.")
            quantize_and_save(orig_model_path=BEST_MODEL_PATH)
        try:
            res = benchmark_model(model_path, split=args.split, num_samples=args.num_samples, cpus=args.cpulimit, device=args.device, use_onnx=False)
            logger.info("Benchmark summary: %s", json.dumps(res, indent=2))
        except Exception as e:
            logger.exception("Benchmark failed: %s", e)
            sys.exit(1)

    if args.onnx_benchmark:
        onnx_path = ONNX_MODEL_PATH
        if not os.path.exists(onnx_path):
            logger.info("ONNX model not found; attempting export first.")
            export_onnx()
        try:
            res = benchmark_model(onnx_path, split=args.split, num_samples=args.num_samples, cpus=args.cpulimit, device="cpu", use_onnx=True)
            logger.info("ONNX Benchmark summary: %s", json.dumps(res, indent=2))
        except Exception as e:
            logger.exception("ONNX benchmark failed: %s", e)
            sys.exit(1)

    if args.run_mqtt:
        if mqtt is None:
            logger.error("paho-mqtt not installed. Install: pip install paho-mqtt")
            sys.exit(1)
        if not os.path.exists(args.model_path):
            logger.info("Model not found at %s; if using .pth this script can attempt to quantize then run.", args.model_path)
            if args.model_path.endswith(".pth"):
                quantize_and_save(orig_model_path=BEST_MODEL_PATH)
        # Determine scaling flags: default is force-scale unless --auto-scale or --no-scale supplied
        force_scale = True
        if args.auto_scale:
            force_scale = False
        if args.no_scale:
            force_scale = False
        inf = MQTTInference(model_path=args.model_path,
                            mqtt_broker=args.mqtt_broker,
                            mqtt_port=args.mqtt_port,
                            mqtt_topic=args.mqtt_topic,
                            predictions_topic=args.mqtt_predictions_topic,
                            health_topic=args.mqtt_health_topic,
                            health_interval=args.health_interval,
                            mqtt_username=args.mqtt_username,
                            mqtt_password=args.mqtt_password,
                            # TLS arguments removed in rollback
                            connect_retries=args.connect_retries,
                            connect_backoff_initial=args.connect_backoff_initial,
                            connect_backoff_max=args.connect_backoff_max,
                            connect_backoff_mult=args.connect_backoff_mult,
                            connect_jitter=args.connect_jitter,
                            log_csv=args.log,
                            cpulimit=args.cpulimit,
                            device=args.device,
                            quantize_on_load=args.quantize_on_load,
                            force_scale=force_scale,
                            no_scale=args.no_scale,
                            debug_features=args.debug_features,
                            feature_order_override=args.feature_order_path,
                            scaler_override=args.scaler_path,
                            evaluation_json_override=args.evaluation_json_path,
                            enable_ensemble=args.enable_ensemble,
                            min_feature_presence=args.min_feature_presence)
        logger.debug("Feature order length=%d", len(inf.feature_order))
        # Retain flag for prediction publishes (dashboard can catch last value on connect)
        try:
            inf._retain_predictions = bool(getattr(args, 'mqtt_retain_predictions', False))
            if inf._retain_predictions:
                logger.info("Prediction publishes will be retained (retain=True)")
        except Exception:
            pass
        try:
            inf._adaptive_mixed_scale = bool(getattr(args, 'adaptive_mixed_scale', False)) and (not args.strict_full_scale)
            if inf._adaptive_mixed_scale:
                logger.info("Adaptive mixed-domain scaling ENABLED")
            inf._auto_reconstruct_mixed = bool(getattr(args, 'auto_reconstruct_mixed', False)) and (not args.strict_full_scale)
            if inf._auto_reconstruct_mixed:
                logger.info("Auto reconstruction of mixed standardized stream ENABLED")
            inf._strict_full_scale = bool(getattr(args, 'strict_full_scale', False))
            if inf._strict_full_scale:
                logger.info("Strict full scaling mode ENABLED (Phase3 style: single scaler.transform, no adaptive/reconstruction)")
            inf._ignore_eval_standardized = bool(getattr(args, 'ignore_eval_standardized', False))
            if inf._ignore_eval_standardized:
                logger.info("Ignoring eval.json already_standardized flag per --ignore-eval-standardized")
            inf._auto_calibrate_temperature = bool(getattr(args, 'auto_calibrate_temperature', False))
            if inf._auto_calibrate_temperature:
                logger.info("Auto temperature calibration ENABLED (monitoring logit gap p99)")
            if getattr(args, 'lock_noscale_if_standardized', False):
                inf._lock_noscale_if_standardized = True
                logger.info("Lock-noscale-if-standardized ENABLED")
            if getattr(args, 'enable_temp_decay', False):
                inf._enable_temp_decay = True
                logger.info("Temperature decay ENABLED")
        except Exception:
            pass
        inf.start()

if __name__ == "__main__":
    main()
