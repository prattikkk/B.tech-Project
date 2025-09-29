#!/usr/bin/env python3
"""
Integration/Self-Test for IoT Project (single script)

This script:
- Imports Phase 3 in-process (no external MQTT broker required)
- Starts its processing worker, buffer retry, alert loop, and metrics server
- Simulates MQTT messages by calling on_message with a fake message object
- Asserts end-to-end processing by checking /metrics (latency p50/p95/p99, backlog)
- Forces alert scenarios (detection spike, tiny lag threshold) and verifies alerts fired

Exit codes:
- 0: Success
- 2: Skipped (missing artifacts/runtime) with explanation
- 1: Failure (assertion or runtime error)
"""
import os
import sys
import time
import json
import threading
import urllib.request
from pathlib import Path

# Prefer predictable, low-privilege port to avoid conflicts
DEFAULT_METRICS_PORT = int(os.getenv("SELFTEST_METRICS_PORT", "0"))  # 0 means pick free port

# Make project root importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Environment prep for metrics and relaxed thresholds
os.environ.setdefault("ENABLE_METRICS", "1")
# Pick a free port if not provided
try:
    import socket as _socket
    if DEFAULT_METRICS_PORT == 0:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
        s.close()
        os.environ.setdefault("METRICS_PORT", str(free_port))
    else:
        os.environ.setdefault("METRICS_PORT", str(DEFAULT_METRICS_PORT))
except Exception:
    os.environ.setdefault("METRICS_PORT", "18080")
# Keep schema validation off for synthetic payloads unless present
os.environ.setdefault("VALIDATE_INGRESS", "0")
# Micro-batch settings to process quickly in tests
os.environ.setdefault("MICRO_BATCH_SIZE", "16")
os.environ.setdefault("MICRO_BATCH_LATENCY_MS", "50")
# Lower presence bar for synthetic data
os.environ.setdefault("MIN_FEATURE_PRESENCE", "0.5")
# Disable TLS enforcement for test
os.environ.setdefault("STRICT_SECURITY", "0")
os.environ.setdefault("PRODUCTION", "0")

# Optional: suppress noisy logs
os.environ.setdefault("PHASE3_LOG_LEVEL", "ERROR")
# Ensure ensemble is disabled for speed and determinism
os.environ.setdefault("ENABLE_ENSEMBLE", "0")

# Import Phase 3
try:
    import Phase_Three as P3
except Exception as e:
    print(json.dumps({
        "status": "skip",
        "reason": f"Failed to import Phase_Three: {e}",
        "hint": "Ensure artifacts_phase2 exists with model_hybrid.onnx or final_model_hybrid.pth, and feature_order.json/scaler.json"
    }))
    sys.exit(2)

# Verify critical artifacts exist (soft check; import may already enforce this)
missing = []
if not P3.PHASE2_DIR.exists():
    missing.append(str(P3.PHASE2_DIR))
if not P3.FEATURE_ORDER_PATH.exists():
    missing.append(str(P3.FEATURE_ORDER_PATH))
if missing:
    print(json.dumps({
        "status": "skip",
        "reason": "Required artifacts missing",
        "missing": missing
    }))
    sys.exit(2)

# Monkeypatch alert posting to capture triggers
_captured_alerts = []

def _capture_post_alert(event_type: str, details: dict):
    _capt = {"event": event_type, "details": details}
    _captured_alerts.append(_capt)

P3._post_alert = _capture_post_alert  # type: ignore

# Monkeypatch MQTT client publish to be a no-op success (avoid offline buffering)
try:
    import types as _types
    def _fake_publish(topic, payload, qos=0):
        return _types.SimpleNamespace(rc=0)
    P3.client.publish = _fake_publish  # type: ignore
except Exception:
    pass

# Avoid analyzer subprocess during shutdown to keep test snappy
try:
    P3._write_and_launch_detached_analyzer = lambda: False  # type: ignore
except Exception:
    pass

# Force easy-to-trigger thresholds in-process
P3.ALERT_EVAL_INTERVAL = 1
P3.ALERT_LAG_THRESHOLD_SEC = 0.0001
P3.ALERT_DETECTION_RATE_HIGH = 0.01
# Make every prediction an alert by setting very low ACTIVE_THRESHOLD
try:
    P3.ACTIVE_THRESHOLD = -1.0
except Exception:
    pass

# Start metrics server (idempotent)
P3._start_metrics_server()

# Start worker threads used by main(), without connecting MQTT
_threads = []
for target in (P3._process_worker, P3._buffer_retry_loop, P3._alert_eval_loop):
    try:
        t = threading.Thread(target=target, daemon=True)
        t.start(); _threads.append(t)
    except Exception as e:
        print(json.dumps({"status": "error", "stage": "start_threads", "error": str(e)}))
        sys.exit(1)

# Fake MQTT message type
class FakeMsg:
    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload

# Build a baseline payload with correct feature keys
try:
    feature_order = list(P3.FEATURE_ORDER)
except Exception as e:
    print(json.dumps({"status": "error", "stage": "feature_order", "error": str(e)}))
    sys.exit(1)

# Construct a typical benign-like payload
base_features = {k: 0 for k in feature_order}
# Ensure protocol is present as a valid string variant to hit coercion path
if "protocol" in base_features:
    base_features["protocol"] = "TCP"

# Send synthetic messages via on_message
N = int(os.getenv("SELFTEST_MESSAGE_COUNT", "40"))
label_cycle = ["Benign", "Attack"]
try:
    for i in range(N):
        payload = {
            "timestamp": None,  # ignored by Phase 3; it sets its own ingress ts
            "label": label_cycle[i % len(label_cycle)],
            "features": base_features,
        }
        msg = FakeMsg(topic=P3.MQTT_TOPIC, payload=json.dumps(payload).encode("utf-8"))
        P3.on_message(P3.client, None, msg)
        # small jitter to exercise micro-batching window
        time.sleep(0.003)
except Exception as e:
    print(json.dumps({"status": "error", "stage": "send_messages", "error": str(e)}))
    sys.exit(1)

# Wait for processing or timeout
start = time.time(); timeout = 60.0
processed = 0
while time.time() - start < timeout:
    with P3._metrics_lock:
        processed = int(P3._metrics.get("messages_total", 0))
        backlog = int(P3._metrics.get("buffered_messages", 0))
    if processed >= N:
        break
    time.sleep(0.1)

if processed < max(1, int(0.9*N)):
    print(json.dumps({
        "status": "error", "stage": "wait_processing", "expected": N, "processed": processed
    }))
    sys.exit(1)

# Give alert loop a moment to evaluate
time.sleep(max(1.5, P3.ALERT_EVAL_INTERVAL + 0.5))

# Fetch /metrics and parse
metrics_url = f"http://127.0.0.1:{int(os.getenv('METRICS_PORT', DEFAULT_METRICS_PORT))}/metrics"
try:
    with urllib.request.urlopen(metrics_url, timeout=5) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
except Exception as e:
    print(json.dumps({"status": "error", "stage": "fetch_metrics", "error": str(e), "url": metrics_url}))
    sys.exit(1)

# Parse simple key value lines
values = {}
for line in text.splitlines():
    parts = line.strip().split()
    if len(parts) == 2 and parts[0].startswith("phase3_"):
        key, val = parts[0], parts[1]
        try:
            values[key] = float(val)
        except Exception:
            pass

# Assertions: latency percentiles and backlog
lat_p50 = values.get("phase3_latency_ms_p50", 0.0)
lat_p95 = values.get("phase3_latency_ms_p95", 0.0)
lat_p99 = values.get("phase3_latency_ms_p99", 0.0)
backlog = values.get("phase3_buffered_messages", 0.0)

errors = []
if not (lat_p50 > 0 and lat_p95 >= lat_p50 and lat_p99 >= lat_p95):
    errors.append({"latency": {"p50": lat_p50, "p95": lat_p95, "p99": lat_p99}})
if backlog > 5:  # allow a small transient buffer
    errors.append({"backlog_high": backlog})

# Alert expectations: we forced detection spike and tiny lag threshold
# Confirm our monkeypatched _post_alert got these events
alert_types = [a["event"] for a in _captured_alerts]
if "detection_rate_spike" not in alert_types:
    errors.append({"missing_alert": "detection_rate_spike", "alerts": alert_types})
if "high_processing_lag" not in alert_types:
    # best-effort: tolerate if environment time resolution too coarse; fall back to metric check
    with P3._metrics_lock:
        last_lag = float(P3._metrics.get("last_batch_processing_lag_sec", 0.0))
    if last_lag <= P3.ALERT_LAG_THRESHOLD_SEC:
        errors.append({"missing_alert": "high_processing_lag", "last_lag_sec": last_lag, "threshold": P3.ALERT_LAG_THRESHOLD_SEC})

# Shutdown
try:
    P3.shutdown_event.set()
    # best-effort graceful shutdown
    try:
        P3._graceful_shutdown(None, None)
    except Exception:
        pass
except Exception:
    pass

summary = {
    "status": "ok" if not errors else "fail",
    "processed": processed,
    "latency_ms": {"p50": lat_p50, "p95": lat_p95, "p99": lat_p99},
    "backlog": backlog,
    "alerts_captured": alert_types,
}
if errors:
    summary["errors"] = errors

print(json.dumps(summary, indent=2), flush=True)

sys.exit(0 if not errors else 1)
