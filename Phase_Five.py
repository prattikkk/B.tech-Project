#!/usr/bin/env python3
"""
Phase_Five.py - Streamlit Live IoT Anomaly Detection Dashboard

The dashboard connects to the MQTT broker to receive predictions from Phase 4
and displays real-time anomaly detection results with live charts and alerts.
"""

import os
import json
import time
import queue
from datetime import datetime
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from rotating_csv_logger import RotatingCSVLogger
except Exception:
    RotatingCSVLogger = None  # type: ignore

# MQTT client for receiving predictions
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    st.error("⚠️ paho-mqtt not installed. Install with: pip install paho-mqtt")

# ------------- Configuration -------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_PREDICTIONS_TOPIC", "iot/traffic/predictions")
HEALTH_TOPIC = os.getenv("MQTT_HEALTH_TOPIC", "iot/traffic/health")
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.495))  # From Phase 2 evaluation

# UI update configuration
POLL_INTERVAL = 0.1         # seconds between UI updates
MAX_POINTS = 300           # max points in chart
MAX_BATCH = 50             # max messages per update cycle
TABLE_ROWS = 15            # rows shown in table
EMA_ALPHA = 0.2            # smoothing factor for EMA curve
EVAL_JSON_PATH = os.getenv("EVALUATION_JSON_PATH", os.path.join(os.path.dirname(__file__), "artifacts_phase2", "evaluation.json"))
_EVAL_INFO = {}
try:
    if os.path.exists(EVAL_JSON_PATH):
        with open(EVAL_JSON_PATH, 'r', encoding='utf-8') as _fh:
            _j = json.load(_fh)
        _temp = float(_j.get('calibration', {}).get('temperature', 1.0))
        _thr_full = float(_j.get('meta', {}).get('val_best_threshold', 0.5))
        _thr_deep = float(_j.get('meta', {}).get('val_best_threshold_deep_only', _thr_full))
        _ens = bool(_j.get('ensemble', {}).get('use_lgbm', False))
        if ANOMALY_THRESHOLD <= 0.0:
            ANOMALY_THRESHOLD = _thr_full if _ens else _thr_deep
        _EVAL_INFO = {
            'temperature_eval': _temp,
            'thr_full': _thr_full,
            'thr_deep': _thr_deep,
            'ensemble_eval': _ens,
            'active_initial': ANOMALY_THRESHOLD,
        }
    else:
        _EVAL_INFO = {'warning': 'evaluation.json missing'}
except Exception as _e:
    _EVAL_INFO = {'error': str(_e)}

# ------------- App State (Singleton) -------------
class DashboardState:
    def __init__(self, maxlen=MAX_POINTS):
        self.queue = queue.Queue(maxsize=5000)
        self.client = None
        self.connected = False
        self.auto_started = False
        self.next_reconnect_ts = 0.0

        # Time series data for chart
        self.times = deque(maxlen=maxlen)
        self.probs = deque(maxlen=maxlen)
        self.predictions = deque(maxlen=maxlen)
        self.ema_probs = deque(maxlen=maxlen)

        # Recent messages for table
        self.recent_messages = deque(maxlen=maxlen)

        # Alerts
        self.alerts = deque(maxlen=100)

        # Statistics
        self.total_messages = 0
        self.total_attacks = 0
        self.total_benign = 0
        self.total_alerts = 0
        # Latency & timing
        self.last_msg_received_ts = None  # wall-clock datetime when processed
        self.last_msg_timestamp_field = None  # timestamp string from message
        self.last_inference_ms = None

        # Health metrics buffers
        self.health_samples = deque(maxlen=500)
        self.last_health = None
        self.guardrail_history = deque(maxlen=200)
        self.psi_history = deque(maxlen=200)
        self.mean_shift_history = deque(maxlen=200)
        self.std_ratio_history = deque(maxlen=200)
        # Optional local prediction logging
        self.pred_log_enabled = os.getenv("DASHBOARD_LOG_PREDICTIONS", "0").lower() in ("1","true","yes")
        self._pred_logger = None
        if self.pred_log_enabled and RotatingCSVLogger is not None:
            try:
                log_path = Path(os.getenv("DASHBOARD_LOG_PATH", os.path.join(os.path.dirname(__file__), "artifacts_phase2", "phase5_dashboard_log.csv")))
                self._pred_logger = RotatingCSVLogger(log_path)
            except Exception:
                self.pred_log_enabled = False

@st.cache_resource
def get_dashboard_state():
    state = DashboardState(maxlen=MAX_POINTS)
    # Attach a small error ring buffer for thread-safe reporting via main loop
    state.errors = deque(maxlen=50)
    return state

# ------------- MQTT Client Functions -------------
def on_connect(client, userdata, flags, rc, properties=None):
    """MQTT connection callback (no Streamlit calls here; update state only)"""
    try:
        if rc == 0:
            userdata.connected = True
            client.subscribe(MQTT_TOPIC)
            if getattr(userdata, 'health_topic', None):
                client.subscribe(userdata.health_topic)
        else:
            userdata.connected = False
            # record error for display in main loop
            if hasattr(userdata, 'errors'):
                userdata.errors.append(f"Connect failed: rc={rc}")
    except Exception:
        pass

def on_disconnect(client, userdata, *args, **kwargs):
    """Robust disconnect handler that tolerates v3/v5 signature variants.

    paho-mqtt may invoke as:
      v3 style: on_disconnect(client, userdata, rc)
      v5 style: on_disconnect(client, userdata, flags, rc, properties=None)
    We normalize by extracting the last positional int as rc.
    """
    try:
        rc = None
        # args could be (rc,) or (flags, rc, properties)
        if args:
            # rc should be last int in args
            for a in reversed(args):
                if isinstance(a, int):
                    rc = a; break
        if rc is None:
            rc = kwargs.get('rc', -1)
        userdata.connected = False
        userdata.next_reconnect_ts = time.time() + 2
        if hasattr(userdata, 'errors'):
            userdata.errors.append(f"Disconnected rc={rc}")
    except Exception:
        pass

def on_message(client, userdata, msg):
    """MQTT message callback - receive predictions and enqueue; avoid Streamlit calls here."""
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        if topic == HEALTH_TOPIC:
            # Phase 4 health heartbeat
            hb = {
                'ts': payload.get('timestamp') or payload.get('ts') or datetime.utcnow().isoformat(),
                'rolling_accuracy': payload.get('rolling_accuracy'),
                'baseline_acc': payload.get('baseline_acc'),
                'guardrail_status': payload.get('guardrail_status'),
                'guardrail_streak': payload.get('guardrail_streak'),
                'recalibration_flag': payload.get('recalibration_flag'),
                'psi_mean': payload.get('psi_mean'),
                'mean_shift_avg': payload.get('mean_shift_avg'),
                'std_ratio_avg': payload.get('std_ratio_avg'),
            }
            try:
                userdata.health_samples.append(hb)
                userdata.last_health = hb
                if hb['guardrail_status'] is not None:
                    userdata.guardrail_history.append(hb['guardrail_status'])
                if hb['psi_mean'] is not None:
                    userdata.psi_history.append(hb['psi_mean'])
                if hb['mean_shift_avg'] is not None:
                    userdata.mean_shift_history.append(hb['mean_shift_avg'])
                if hb['std_ratio_avg'] is not None:
                    userdata.std_ratio_history.append(hb['std_ratio_avg'])
            except Exception:
                pass
            return
        # Unified prediction format from Phase 3/4 publisher
        timestamp = payload.get("timestamp", datetime.utcnow().isoformat())
        true_label = payload.get("true_label", -1)
        prediction = payload.get("pred", 0)
        prob_attack = payload.get("prob_attack", 0.0)
        # Append debug meta if present
        threshold_meta = payload.get('threshold')
        temperature_meta = payload.get('temperature')
        ensemble_meta = payload.get('ensemble_enabled')
        message_data = {"timestamp": timestamp, "true_label": true_label, "prediction": prediction, "prob_attack": prob_attack}
        if threshold_meta is not None:
            message_data['threshold'] = threshold_meta
        if temperature_meta is not None:
            message_data['temperature'] = temperature_meta
        if ensemble_meta is not None:
            message_data['ensemble_enabled'] = ensemble_meta
        try:
            userdata.queue.put_nowait(message_data)
        except queue.Full:
            try:
                userdata.queue.get_nowait()
                userdata.queue.put_nowait(message_data)
            except Exception:
                pass
        # Optional rotating log write
        try:
            if getattr(userdata, 'pred_log_enabled', False) and getattr(userdata, '_pred_logger', None) is not None:
                userdata._pred_logger.log(datetime.utcnow().isoformat(), true_label, prediction, prob_attack)
        except Exception:
            pass
    except Exception as e:
        try:
            if hasattr(userdata, 'errors'):
                userdata.errors.append(f"Parse error: {e}")
        except Exception:
            pass

def start_mqtt_client(state):
    """Start MQTT client to receive predictions from Phase 4"""
    if not MQTT_AVAILABLE:
        return False, "paho-mqtt not installed"
        
    if state.client is not None:
        return True, "MQTT client already running"
        
    try:
        client = mqtt.Client(
            client_id=f"phase5-dashboard-{int(time.time())}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        client.user_data_set(state)
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        # enable auto reconnect with backoff
        try:
            client.reconnect_delay_set(min_delay=1, max_delay=30)
        except Exception:
            pass
        
        # use async connect so we don't block the UI
        client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
        # Subscribe to health topic after connect via on_connect hook; add fallback timer
        # We'll piggyback in on_connect but track second topic here for clarity
        # (on_connect does subscribe to MQTT_TOPIC already; we store health topic for later)
        state.health_topic = HEALTH_TOPIC
        client.loop_start()
        state.client = client
        return True, "MQTT client started"
    except Exception as e:
        return False, f"Failed to start MQTT client: {e}"

def stop_mqtt_client(state):
    """Stop MQTT client"""
    if state.client:
        try:
            state.client.loop_stop()
            state.client.disconnect()
        except Exception:
            pass
        finally:
            state.client = None
            state.connected = False

# ------------- Data Processing -------------
def process_message_queue(state, max_batch=MAX_BATCH):
    """Process incoming messages from MQTT queue"""
    processed = 0
    
    while processed < max_batch:
        try:
            message = state.queue.get_nowait()
        except queue.Empty:
            break
            
        # Parse timestamp
        try:
            if isinstance(message["timestamp"], str):
                ts = datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
            else:
                ts = datetime.utcnow()
        except:
            ts = datetime.utcnow()
            
        # Extract data
        prob_attack = float(message.get("prob_attack", 0.0))
        # Capture inference latency if provided
        if "inference_ms" in message:
            try:
                state.last_inference_ms = float(message.get("inference_ms"))
            except Exception:
                pass
        dyn_thr = message.get('threshold')
        dyn_temp = message.get('temperature')
        dyn_ens = message.get('ensemble_enabled')
        if dyn_thr is not None:
            global ANOMALY_THRESHOLD
            ANOMALY_THRESHOLD = float(dyn_thr)
            if hasattr(state, 'errors'):
                state.errors.append(f"Threshold-> {ANOMALY_THRESHOLD:.3f}")
        if dyn_temp is not None:
            state.last_temperature = float(dyn_temp)
        if dyn_ens is not None:
            state.last_ensemble = bool(dyn_ens)
        prediction = int(message.get("prediction", 0))
        true_label = message.get("true_label", -1)
        
        # Update time series
        state.times.append(ts)
        state.probs.append(prob_attack)
        state.predictions.append(prediction)
        # Update EMA smoothing
        if len(state.ema_probs) == 0:
            state.ema_probs.append(prob_attack)
        else:
            prev = state.ema_probs[-1]
            state.ema_probs.append(EMA_ALPHA * prob_attack + (1 - EMA_ALPHA) * prev)
        
        # Update recent messages table
        state.recent_messages.appendleft({
            "Time": ts.strftime("%H:%M:%S"),
            "Prediction": "Attack" if prediction == 1 else "Benign",
            "Probability": f"{prob_attack:.4f}",
            "True Label": "Attack" if true_label == 1 else "Benign" if true_label == 0 else "Unknown"
        })
        
        # Update statistics
        state.total_messages += 1
        if prediction == 1:
            state.total_attacks += 1
        else:
            state.total_benign += 1

        # Track timing
        state.last_msg_received_ts = datetime.utcnow()
        state.last_msg_timestamp_field = message.get("timestamp", None)
            
        # Check for alerts (high probability attacks)
        if prob_attack >= ANOMALY_THRESHOLD:
            state.alerts.appendleft({
                "time": ts.strftime("%H:%M:%S"),
                "probability": prob_attack,
                "severity": "HIGH" if prob_attack >= 0.8 else "MEDIUM"
            })
            state.total_alerts += 1
            
        processed += 1
        
    return processed

# ------------- Streamlit UI -------------
st.set_page_config(page_title="IoT Anomaly Detection Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Anomaly Detection - Live Dashboard")
st.caption("Live predictions via MQTT from Phase 3/4")

# Get app state
state = get_dashboard_state()

# Sidebar Controls
st.sidebar.header("🔧 Controls")

# MQTT Connection
if st.sidebar.button("🔌 Connect MQTT"):
    success, message = start_mqtt_client(state)
    (st.sidebar.success if success else st.sidebar.error)(message)

if st.sidebar.button("🔌 Disconnect MQTT"):
    stop_mqtt_client(state)
    st.sidebar.warning("MQTT client disconnected")

if st.sidebar.button("🗑️ Clear Data"):
    state.times.clear()
    state.probs.clear()
    state.predictions.clear()
    state.recent_messages.clear()
    state.alerts.clear()
    state.total_messages = 0
    state.total_attacks = 0
    state.total_benign = 0
    state.total_alerts = 0
    st.sidebar.success("Data cleared")

with st.sidebar.expander("⚙️ Configuration", expanded=False):
    st.text(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    st.text(f"Topic: {MQTT_TOPIC}")
    st.text(f"Threshold: {ANOMALY_THRESHOLD}")

# Connection Status
connection_status = "🟢 Connected" if state.connected else "🔴 Disconnected"
st.sidebar.metric("Connection Status", connection_status)

# Compact stats
stats_expander = st.sidebar.expander("📊 Stats", expanded=False)
with stats_expander:
    stats_placeholder = st.empty()

# Main Dashboard Layout (simplified)
col_chart, col_side = st.columns([3, 1])

with col_chart:
    st.subheader("📈 Attack Probability")
    chart_placeholder = st.empty()
    st.subheader("📋 Recent Predictions")
    table_placeholder = st.empty()

with col_side:
    st.subheader("🎯 Threat Level")
    threat_placeholder = st.empty()
    st.subheader("🚨 Alerts")
    alerts_placeholder = st.empty()
    st.subheader("🛡️ Guardrail & Drift")
    guardrail_placeholder = st.empty()
    drift_chart_placeholder = st.empty()
    st.subheader("⏱️ Timing")
    timing_placeholder = st.empty()

# Auto-start MQTT once per session
if not state.auto_started and state.client is None:
    ok, msg = start_mqtt_client(state)
    state.auto_started = True

# Main update loop
status_placeholder = st.empty()

try:
    while True:
        # Process new messages
        new_messages = process_message_queue(state, MAX_BATCH)
        
        # Build chart data (with EMA smoothing and threshold line)
        if len(state.probs) > 0:
            times = list(state.times)
            probs = list(state.probs)
            ema = list(state.ema_probs) if len(state.ema_probs) == len(probs) else probs
            df = pd.DataFrame({
                "Time": times,
                "Probability": probs,
                "EMA": ema,
                "Threshold": [ANOMALY_THRESHOLD] * len(times)
            })
            try:
                import altair as alt
                line_prob = alt.Chart(df).mark_line(color="#5B9BD5").encode(x="Time:T", y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0,1])), tooltip=["Time:T","Probability:Q"])
                line_ema = alt.Chart(df).mark_line(color="#ED7D31").encode(x="Time:T", y="EMA:Q")
                rule_thr = alt.Chart(df).mark_rule(color="#9C0006", strokeDash=[4,4]).encode(y="Threshold:Q")
                chart = (line_prob + line_ema + rule_thr).properties(height=260)
                chart_placeholder.altair_chart(chart, use_container_width=True)
            except Exception:
                chart_placeholder.line_chart(df.set_index("Time")[ ["Probability"] ])
        
        # Update recent messages table
        if state.recent_messages:
            table_df = pd.DataFrame(list(state.recent_messages)[:TABLE_ROWS])
            table_placeholder.dataframe(table_df, use_container_width=True, height=400)
        else:
            table_placeholder.info("⏳ Waiting for predictions...")
        
        # Update alerts
        if state.alerts:
            with alerts_placeholder.container():
                last_alerts = list(state.alerts)[:5]  # compact view
                for alert in last_alerts:
                    severity_color = "🔴" if alert["severity"] == "HIGH" else "🟡"
                    st.write(f"{severity_color} {alert['time']} — p={alert['probability']:.3f}")
        else:
            alerts_placeholder.info("✅ No recent security alerts")
        
        # Update threat level
        current_threat = "LOW"
        if state.total_messages > 0:
            recent_attacks = sum(1 for p in list(state.probs)[-20:] if p >= ANOMALY_THRESHOLD)
            if recent_attacks >= 10:
                current_threat = "HIGH"
            elif recent_attacks >= 5:
                current_threat = "MEDIUM"
        
        threat_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[current_threat]
        threat_placeholder.metric("Current Level", f"{threat_color} {current_threat}")
        suffix = f" Thr={ANOMALY_THRESHOLD:.3f}"
        if hasattr(state, 'last_temperature'):
            suffix += f" T={state.last_temperature:.2f}"
        if hasattr(state, 'last_ensemble'):
            suffix += " ENS:on" if state.last_ensemble else " ENS:off"
        threat_placeholder.metric("Current Level", f"{threat_color} {current_threat}", help=suffix)
        
        # Guardrail & drift visualization
        try:
            hb = state.last_health
            if hb:
                # Guardrail status indicator
                status_map = {0: ('Stable','🟢'), 1: ('Notice','🟡'), 2: ('Alert','🔴')}
                label, icon = status_map.get(hb.get('guardrail_status',0), ('Unknown','⚪'))
                rolling_acc = hb.get('rolling_accuracy')
                baseline = hb.get('baseline_acc')
                if rolling_acc is not None and baseline:
                    pct_of_baseline = (rolling_acc / baseline * 100) if baseline else None
                else:
                    pct_of_baseline = None
                recal_flag = hb.get('recalibration_flag')
                guardrail_md = f"{icon} **Guardrail:** {label}"
                if rolling_acc is not None:
                    guardrail_md += f" | Rolling: {rolling_acc:.3f}"
                if baseline:
                    guardrail_md += f" (Baseline: {baseline:.3f})"
                if pct_of_baseline is not None:
                    guardrail_md += f" ({pct_of_baseline:.1f}% of baseline)"
                if recal_flag:
                    guardrail_md += " \n🔄 Recalibration Flag: TRUE"
                guardrail_placeholder.markdown(guardrail_md)
                # Drift mini-table / chart
                if state.psi_history:
                    df_drift = pd.DataFrame({
                        'idx': range(len(state.psi_history)),
                        'psi_mean': list(state.psi_history),
                        'mean_shift': list(state.mean_shift_history) if state.mean_shift_history else [np.nan]*len(state.psi_history),
                        'std_ratio': list(state.std_ratio_history) if state.std_ratio_history else [np.nan]*len(state.psi_history),
                    })
                    try:
                        import altair as alt
                        base = alt.Chart(df_drift).transform_fold(
                            ['psi_mean','mean_shift','std_ratio'],
                            as_=['metric','value']
                        ).mark_line().encode(
                            x='idx:Q', y=alt.Y('value:Q', title='Value'), color='metric:N'
                        ).properties(height=160)
                        drift_chart_placeholder.altair_chart(base, use_container_width=True)
                    except Exception:
                        drift_chart_placeholder.dataframe(df_drift.tail(50), use_container_width=True)
            else:
                guardrail_placeholder.info("Awaiting health data...")
        except Exception:
            guardrail_placeholder.warning("Error rendering guardrail/drift section")
        
        # Show connection info
        if state.connected:
            status_placeholder.info(f"📡 Connected — queue {state.queue.qsize()} | {datetime.now().strftime('%H:%M:%S')}")
        else:
            status_placeholder.warning("📡 Reconnecting to MQTT broker...")
            # manual reconnect guard in case auto-reconnect isn't active yet
            if state.client and time.time() >= state.next_reconnect_ts:
                try:
                    state.client.reconnect()
                    state.next_reconnect_ts = time.time() + 5
                except Exception:
                    state.next_reconnect_ts = time.time() + 5

        # Timing panel
        try:
            if state.last_msg_received_ts:
                age_sec = (datetime.utcnow() - state.last_msg_received_ts).total_seconds()
                inf_ms = state.last_inference_ms
                msg_ts = state.last_msg_timestamp_field
                timing_md = f"**Last Msg Age:** {age_sec:.2f}s"
                if inf_ms is not None:
                    timing_md += f"  |  **Inference:** {inf_ms:.1f} ms"
                if msg_ts:
                    # show truncated ISO
                    timing_md += f"\n`ts={str(msg_ts)[:23]}`"
                timing_placeholder.markdown(timing_md)
            else:
                timing_placeholder.info("No messages yet")
        except Exception:
            timing_placeholder.warning("Timing calc error")
        
        # Show recent errors (if any)
        if getattr(state, 'errors', None):
            # Display only the last 3 errors to avoid noise
            for err in list(state.errors)[-3:]:
                status_placeholder.warning(f"⚠️ {err}")

    # Update compact sidebar stats
    stats_md = f"Msgs: {state.total_messages} | Attacks: {state.total_attacks} | Benign: {state.total_benign} | Alerts: {state.total_alerts}"
    stats_placeholder.caption(stats_md)

    # Sleep before next update
    time.sleep(POLL_INTERVAL)
        
except KeyboardInterrupt:
    st.info("Dashboard stopped by user")
except Exception as e:
    st.error(f"Dashboard error: {e}")
finally:
    # Cleanup
    stop_mqtt_client(state)
