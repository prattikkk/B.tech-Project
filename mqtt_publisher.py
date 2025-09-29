#!/usr/bin/env python3
"""
MQTT Publisher (enhanced): realistic IoT traffic generator from Phase-1 artifacts

Publishes dataset rows as JSON payloads to an MQTT topic, with options to:
 - nest features under {"features": {...}}
 - unscale scaled data back to raw-like values
 - simulate bursty traffic, jitter, duplicates, missing fields, and out-of-order delivery
 - emit protocol as human-readable name (TCP/UDP/ICMP) to test mapping on the consumer

Usage example:
    python mqtt_publisher.py --base-dir "C:\\Users\\prati\\Project" --split test \
            --n-msgs 1000 --rate 20 --nested --unscale --bursty --burst-size 40 --burst-sleep 1.5 \
            --drop-rate 0.05 --dup-rate 0.01 --ooo-rate 0.02 --protocol-as-name
"""
import argparse
import json
import time
import random
import pickle
import joblib
import sys
import signal
import numpy as np
import paho.mqtt.client as mqtt
import logging
from datetime import datetime
from collections import deque
from pathlib import Path

# --- Helpers ---
def load_data_from_artifacts(base_dir: Path):
    p_npz = base_dir / "artifacts_phase1" / "data.npz"
    p_pkl = base_dir / "artifacts_phase1" / "data.pkl"
    if p_npz.exists():
        try:
            npz = np.load(p_npz, allow_pickle=True)
            data = {
                "X_train": npz["X_train"], "y_train": npz["y_train"],
                "X_val":   npz["X_val"],   "y_val":   npz["y_val"],
                "X_test":  npz["X_test"],  "y_test":  npz["y_test"]
            }
            logging.info("Loaded data from %s", p_npz)
            return data
        except Exception as e:
            logging.warning("Failed to use .npz %s -- falling back to pkl if present", e)

    if p_pkl.exists():
        with open(p_pkl, "rb") as fh:
            data = pickle.load(fh)
        logging.info("Loaded data from %s", p_pkl)
        return data

    raise FileNotFoundError("No data.npz or data.pkl found in artifacts_phase1")

def load_feature_order(base_dir: Path, prefer_phase2=True):
    p2 = base_dir / "artifacts_phase2" / "feature_order.json"
    p1 = base_dir / "artifacts_phase1" / "feature_order.json"
    keep_idx_path = base_dir / "artifacts_phase2" / "evaluation.json"
    if prefer_phase2 and p2.exists():
        fo = json.loads(p2.read_text())
    elif p1.exists():
        fo = json.loads(p1.read_text())
    else:
        raise FileNotFoundError("feature_order.json not found in phase1 or phase2 artifacts")
    kept_idx = None
    # If evaluation.json contains kept_feature_indices, apply them
    if keep_idx_path.exists():
        try:
            ej = json.loads(keep_idx_path.read_text())
            kept_idx = ej.get("meta", {}).get("kept_feature_indices", None)
            if kept_idx:
                fo = [fo[i] for i in kept_idx]
                # Important: return None for kept_idx now to avoid re-applying downstream (was causing misalignment)
                logging.info("Applied kept_feature_indices once (feature count -> %d)", len(fo))
                kept_idx = None
        except Exception:
            pass
    return fo, kept_idx

def load_scaler(base_dir: Path):
    # Support scaler.pkl (joblib) or scaler.json exported with mean/scale
    p_pkl = base_dir / "artifacts_phase1" / "scaler.pkl"
    p_json = base_dir / "artifacts_phase1" / "scaler.json"
    if p_json.exists():
        try:
            j = json.loads(p_json.read_text())
            mean = np.array(j.get("mean", []), dtype=np.float32)
            scale = np.array(j.get("scale", []), dtype=np.float32)
            return {"type": "json", "mean": mean, "scale": scale}
        except Exception:
            pass
    if p_pkl.exists():
        try:
            obj = joblib.load(p_pkl)
            # StandardScaler-like with mean_ and scale_ attributes
            mean = getattr(obj, "mean_", None)
            scale = getattr(obj, "scale_", None)
            if mean is not None and scale is not None:
                return {"type": "pkl", "obj": obj, "mean": np.array(mean, dtype=np.float32), "scale": np.array(scale, dtype=np.float32)}
            # else return object anyway
            return {"type": "pkl", "obj": obj}
        except Exception as e:
            logging.warning("Could not load scaler.pkl: %s", e)
    return None

def unscale_row(row_scaled, scaler_meta):
    """Given a 1-D scaled row, return raw-like row using scaler metadata."""
    if scaler_meta is None:
        return row_scaled
    if scaler_meta["type"] == "json":
        mean = scaler_meta["mean"]
        scale = scaler_meta["scale"]
        if mean.size == 0 or scale.size == 0:
            return row_scaled
        return (row_scaled * scale) + mean
    else:
        if "obj" in scaler_meta:
            s = scaler_meta["obj"]
            # If it has mean_ and scale_ we can use them
            if hasattr(s, "mean_") and hasattr(s, "scale_"):
                return (row_scaled * np.array(s.scale_)) + np.array(s.mean_)
        # fallback: if we don't know how, return as-is
        return row_scaled

# --- Graceful stop ---
stop_requested = False
def _sigint(sig, frame):
    global stop_requested
    stop_requested = True
    logging.info("Stop requested, finishing...")
signal.signal(signal.SIGINT, _sigint)
signal.signal(signal.SIGTERM, _sigint)

# ----------------- Main publisher -----------------
def main(args):
    base_dir = Path(args.base_dir)
    data = load_data_from_artifacts(base_dir)
    feature_order, kept_idx = load_feature_order(base_dir, prefer_phase2=True)
    scaler_meta = load_scaler(base_dir) if args.unscale else None

    # Choose requested split
    split = args.split.lower()
    if split == "val":
        X = data["X_val"]; y = data["y_val"]
    elif split == "test":
        X = data["X_test"]; y = data["y_test"]
    else:
        X = data["X_train"]; y = data["y_train"]

    # Convert to numpy in case they're other types
    X = np.asarray(X)
    y = np.asarray(y)

    # N messages handling
    if args.n_msgs is None or args.n_msgs <= 0:
        N = X.shape[0]
        infinite = False if args.once else True
    else:
        N = min(args.n_msgs, X.shape[0])
        infinite = False

    # Choice of indices: sampling or sequential
    rng = np.random.RandomState(args.seed)
    if args.shuffle:
        indices = rng.choice(X.shape[0], N, replace=False)
    else:
        indices = np.arange(N)

    # MQTT setup
    client = mqtt.Client()
    if args.username:
        client.username_pw_set(args.username, args.password)
    # TLS placeholder
    if args.tls:
        client.tls_set()  # user may want to configure certs; kept minimal here

    try:
        client.connect(args.broker, args.port, keepalive=60)
    except Exception as e:
        logging.error("Could not connect to broker: %s", e)
        sys.exit(2)
    client.loop_start()

    logging.info("Publishing %s messages -> topic=%s rate=%s/s nested_features=%s include_label=%s", N if not infinite else '∞', args.topic, args.rate, args.nested, args.include_label)
    sleep_time = 1.0 / max(0.001, args.rate)

    # For protocol-as-name simulation (Phase 1 mapping: ICMP=1, TCP=6, UDP=17, OTHER=0)
    def _prot_name_from_value(val: float | int):
        try:
            iv = int(val)
        except Exception:
            return "OTHER"
        return {1: "ICMP", 6: "TCP", 17: "UDP"}.get(iv, "OTHER")

    def _maybe_drop_features(feat_dict: dict) -> dict:
        if args.drop_rate <= 0:
            return feat_dict
        out = dict(feat_dict)
        for k in list(out.keys()):
            if random.random() < args.drop_rate:
                out.pop(k, None)
        return out

    def make_payload_from_row(row_raw, label=None):
        # row_raw expected to be raw-like features in same order as feature_order
        payload_features = {feat: float(row_raw[i]) for i, feat in enumerate(feature_order)}
        # Optionally emit protocol as name instead of integer
        if args.protocol_as_name and "protocol" in payload_features:
            payload_features["protocol"] = _prot_name_from_value(payload_features["protocol"])
        # Randomly drop some features to simulate partial data
        payload_features = _maybe_drop_features(payload_features)
        payload = {}
        if args.nested:
            payload["features"] = payload_features
        else:
            payload.update(payload_features)
        if args.include_label and label is not None:
            payload["label"] = int(label)
        payload["device_id"] = f"dev_{random.randint(1, args.device_count)}"
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return payload

    sent = 0
    idx_pointer = 0
    # For out-of-order simulation: small queue that can delay some messages
    delayed_queue = deque()
    while not stop_requested and (infinite or sent < N):
        # pick next index
        if args.shuffle:
            idx = int(indices[sent % len(indices)])
        else:
            idx = int(indices[idx_pointer % len(indices)])
            idx_pointer += 1

        x_row = X[idx].reshape(-1)

        # If X contains scaled data, unscale back to raw-like values when scaler metadata is present
        if scaler_meta is not None:
            try:
                raw = unscale_row(x_row, scaler_meta)
            except Exception as e:
                logging.warning("Unscale failed for index %s: %s", idx, e)
                raw = x_row
        else:
            raw = x_row

        # Prepare payload: select correct features using kept_idx if present (for pruned features alignment)
        if kept_idx is not None and len(kept_idx) > 0:
            if len(raw) >= len(kept_idx):
                raw_use = raw[kept_idx]
            else:
                # pad if too short
                raw_use = np.zeros(len(kept_idx), dtype=float)
                raw_use[:len(raw)] = raw
            logging.info("Using kept_idx for raw selection (len=%d)", len(raw_use))
        else:
            # fallback: truncate or pad to feature_order len
            if raw.size < len(feature_order):
                padded = np.zeros(len(feature_order), dtype=float)
                padded[:raw.size] = raw
                raw_use = padded
            else:
                raw_use = raw[:len(feature_order)]

        label = None
        try:
            label = int(y[idx])
        except Exception:
            label = None

        payload = make_payload_from_row(raw_use, label=label if args.include_label else None)

        # Duplicate some messages
        to_send = [payload]
        if args.dup_rate > 0 and random.random() < args.dup_rate:
            to_send.append(payload)

        # Optionally schedule small out-of-order delay for some messages
        for p in to_send:
            if args.ooo_rate > 0 and random.random() < args.ooo_rate:
                delay = random.uniform(0.05, 0.3)
                delayed_queue.append((time.time() + delay, p))
            else:
                try:
                    client.publish(args.topic, json.dumps(p), qos=args.qos, retain=args.retain)
                except Exception as e:
                    logging.warning("Publish failed: %s", e)

        # Flush any due delayed messages
        now = time.time()
        while delayed_queue and delayed_queue[0][0] <= now:
            _, p_due = delayed_queue.popleft()
            try:
                client.publish(args.topic, json.dumps(p_due), qos=args.qos, retain=args.retain)
            except Exception as e:
                logging.warning("Delayed publish failed: %s", e)

        sent += 1
        if sent % max(1, args.report_every) == 0:
            logging.info("Published %d/%s", sent, (N if not infinite else '∞'))

        # if not infinite and we've reached N -> break
        if not infinite and sent >= N:
            break

        # sleep with jitter and optional bursty behavior
        if args.bursty:
            # send 'burst_size' messages quickly, then sleep 'burst_sleep'
            if (sent % max(1, args.burst_size)) == 0:
                time.sleep(max(0.0, args.burst_sleep))
            else:
                jitter = random.uniform(-args.jitter_frac, args.jitter_frac) * sleep_time
                time.sleep(max(0.0, sleep_time + jitter))
        else:
            jitter = random.uniform(-args.jitter_frac, args.jitter_frac) * sleep_time
            time.sleep(max(0.0, sleep_time + jitter))

    logging.info("Publishing complete or stopped; disconnecting...")
    client.loop_stop()
    client.disconnect()

# ---------------- CLI ----------------
def build_parser():
    p = argparse.ArgumentParser(description="MQTT publisher from Phase1/Phase2 artifacts")
    p.add_argument("--base-dir", type=str, default=".")
    p.add_argument("--split", type=str, choices=["train","val","test"], default="test")
    p.add_argument("--n-msgs", type=int, default=500, help="Total messages to send (None for dataset size)")
    p.add_argument("--rate", type=float, default=20.0, help="messages per second")
    p.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    p.add_argument("--shuffle", action="store_true", help="sample random indices")
    p.add_argument("--include-label", dest="include_label", action="store_true", default=True, help="include label in payload (default=True for testing)")
    p.add_argument("--nested", action="store_true", help="nest features under 'features' key (Phase3 expects this option)")
    p.add_argument("--unscale", action="store_true", help="attempt to unscale values using scaler.pkl/scaler.json")
    p.add_argument("--broker", type=str, default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", type=str, default="iot/traffic")
    p.add_argument("--qos", type=int, default=0, choices=[0,1,2])
    p.add_argument("--retain", action="store_true", help="publish retained messages")
    p.add_argument("--rate-jitter", type=float, dest="jitter_frac", default=0.05, help="fractional jitter on inter-message sleep")
    p.add_argument("--bursty", action="store_true", help="simulate bursts: fast batches then pauses")
    p.add_argument("--burst-size", type=int, default=50, help="messages per burst when --bursty is set")
    p.add_argument("--burst-sleep", type=float, default=1.0, help="seconds to sleep after each burst when --bursty is set")
    p.add_argument("--drop-rate", type=float, default=0.0, help="probability to drop each feature key from a payload")
    p.add_argument("--dup-rate", type=float, default=0.0, help="probability to duplicate a payload (send twice)")
    p.add_argument("--ooo-rate", type=float, default=0.0, help="probability to delay a payload slightly (0.05-0.3s), causing mild out-of-order delivery")
    p.add_argument("--protocol-as-name", action="store_true", help="emit 'protocol' as name (TCP/UDP/ICMP) instead of numeric code")
    p.add_argument("--device-count", type=int, default=10)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--tls", action="store_true", help="enable tls (placeholder - requires cert config)")
    p.add_argument("--username", type=str, default=None)
    p.add_argument("--password", type=str, default=None)
    p.add_argument("--once", action="store_true", help="publish dataset once and exit (default behavior)")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
