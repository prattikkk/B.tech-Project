# IoT Anomaly Detection Project - Phase 3, 4, 5 Execution Guide

## Prerequisites

**Before running these phases, ensure you have completed:**
1. ✅ Phase 1 (data preprocessing) - creates `artifacts_phase1/`
2. ✅ Phase 2 (model training) - creates `artifacts_phase2/` with trained models

**Verify Phase 2 completion:**
```powershell
# Check if Phase 2 artifacts exist
Test-Path .\artifacts_phase2\final_model_hybrid.pth
Test-Path .\artifacts_phase2\best_model_hybrid.pth
Test-Path .\artifacts_phase2\scaler.pkl
```

**Install additional dependencies for Phases 3-5:**
```powershell
pip install paho-mqtt streamlit jsonschema pydantic voluptuous
```

---

## Phase 3: Real-time MQTT Inference Service

Phase 3 runs a real-time anomaly detection service that subscribes to MQTT topics and performs live inference on IoT traffic data.

### Method 1: Using Helper Script (Recommended)
```powershell
# Basic execution with default settings
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase3.ps1

# With custom MQTT broker settings
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase3.ps1 -Broker "192.168.1.100" -Port 1883 -Topic "iot/traffic"
```

### Method 2: Direct Python Execution
```powershell
# Basic execution (uses localhost MQTT broker)
python .\Phase_Three.py

# With environment variables for MQTT configuration
$env:MQTT_BROKER = "localhost"; $env:MQTT_PORT = "1883"; $env:MQTT_TOPIC = "iot/traffic"; python .\Phase_Three.py
```

### Phase 3 Configuration Options

**Environment Variables:**
- `MQTT_BROKER`: MQTT broker hostname (default: localhost)
- `MQTT_PORT`: MQTT broker port (default: 1883)
- `MQTT_TOPIC`: Input topic for IoT traffic data (default: iot/traffic)
- `MQTT_PREDICTIONS_TOPIC`: Output topic for predictions (default: iot/traffic/predictions)
- `MQTT_HEALTH_TOPIC`: Health status topic (default: iot/traffic/health)

**What Phase 3 Does:**
- Loads the trained model from `artifacts_phase2/final_model_hybrid.pth`
- Subscribes to MQTT topics for incoming IoT traffic data
- Performs real-time anomaly detection
- Publishes predictions to output topics
- Maintains health status monitoring

---

## Phase 4: Benchmarking and Edge Deployment

Phase 4 provides multiple deployment options: benchmarking, ONNX export, quantization, and edge inference.

### 4A. Benchmarking (Performance Testing)

```powershell
# Basic benchmark with default settings (2000 samples)
python .\Phase_Four.py --benchmark

# Custom benchmark parameters
python .\Phase_Four.py --benchmark --num_samples 5000 --cpulimit 2

# Using helper script
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase4.ps1

# With custom parameters
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase4.ps1 -NumSamples 5000 -CpuLimit 2
```

### 4B. ONNX Export for Edge Deployment

```powershell
# Export model to ONNX format
python .\Phase_Four.py --export-onnx

# Benchmark ONNX model performance
python .\Phase_Four.py --onnx-benchmark --num_samples 2000 --cpulimit 1
```

### 4C. Quantized Model Testing

```powershell
# Test with dynamic quantization (faster inference)
python .\Phase_Four.py --benchmark --quantize-on-load

# MQTT inference with quantization
python .\Phase_Four.py --run-mqtt --quantize-on-load --mqtt-broker localhost --mqtt-port 1883
```

### 4D. MQTT Edge Inference Service

```powershell
# Run MQTT subscriber with full model
python .\Phase_Four.py --run-mqtt --model .\artifacts_phase2\final_model_hybrid.pth --mqtt-broker localhost --mqtt-port 1883 --mqtt-topic iot/traffic --predictions-topic iot/traffic/predictions --health-topic iot/traffic/health

# Run with quantized model for faster inference
python .\Phase_Four.py --run-mqtt --quantize-on-load --mqtt-broker localhost --mqtt-port 1883
```

### Phase 4 Command Line Options

**Benchmarking:**
- `--benchmark`: Run performance benchmark
- `--num_samples`: Number of samples for benchmark (default: 2000)
- `--cpulimit`: CPU core limit (default: 1)

**ONNX Export:**
- `--export-onnx`: Export model to ONNX format
- `--onnx-benchmark`: Benchmark ONNX model

**MQTT Service:**
- `--run-mqtt`: Start MQTT inference service
- `--mqtt-broker`: MQTT broker hostname
- `--mqtt-port`: MQTT broker port
- `--mqtt-topic`: Input topic for IoT data
- `--predictions-topic`: Output topic for predictions
- `--health-topic`: Health monitoring topic

**Model Options:**
- `--model`: Path to model file (default: artifacts_phase2/final_model_hybrid.pth)
- `--quantize-on-load`: Enable dynamic quantization

---

## Phase 5: Live Dashboard (Streamlit)

Phase 5 provides a web-based dashboard for real-time monitoring of the anomaly detection system.

### Starting the Dashboard

```powershell
# Using helper script (recommended)
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase5.ps1

# Direct streamlit execution
streamlit run .\Phase_Five.py

# With custom port
streamlit run .\Phase_Five.py --server.port 8502
```

### Phase 5 Configuration

**Environment Variables:**
- `MQTT_BROKER`: MQTT broker hostname (default: localhost)
- `MQTT_PORT`: MQTT broker port (default: 1883)
- `MQTT_PREDICTIONS_TOPIC`: Topic for predictions (default: iot/traffic/predictions)
- `MQTT_HEALTH_TOPIC`: Topic for health status (default: iot/traffic/health)
- `ANOMALY_THRESHOLD`: Threshold for anomaly detection (default: 0.495)

**Dashboard Features:**
- Real-time anomaly detection visualization
- Live charts and metrics
- Anomaly alerts and notifications
- System health monitoring
- Historical data trends

---

## Complete Workflow: Running All Phases Together

### Terminal 1: Start MQTT Broker (if needed)
```powershell
# If you don't have an MQTT broker running, install Mosquitto:
# Download from: https://mosquitto.org/download/
# Or use Docker: docker run -it -p 1883:1883 eclipse-mosquitto

# For testing, you can use a public broker or install locally
```

### Terminal 2: Start Phase 3 (Real-time Inference)
```powershell
cd c:\Users\prati\IOTProject
python .\Phase_Three.py
```

### Terminal 3: Start Phase 5 (Dashboard)
```powershell
cd c:\Users\prati\IOTProject
streamlit run .\Phase_Five.py
```

### Terminal 4: Generate Test Traffic (Optional)
```powershell
# Publish test data to drive the system
python .\mqtt_publisher.py --base-dir . --split test --n-msgs 1000 --rate 20 --nested --unscale --protocol-as-name
```

### Terminal 5: Run Phase 4 Benchmarks
```powershell
# Run performance benchmarks
python .\Phase_Four.py --benchmark --num_samples 5000

# Export to ONNX and benchmark
python .\Phase_Four.py --export-onnx
python .\Phase_Four.py --onnx-benchmark --num_samples 2000
```

---

## Accessing the Dashboard

Once Phase 5 is running, access the dashboard at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501

The dashboard will show:
- Real-time anomaly detection results
- Live performance metrics
- System health status
- Interactive charts and alerts

---

## Troubleshooting

### Common Issues:

**1. MQTT Connection Issues:**
```powershell
# Check if MQTT broker is running
# For Windows with Mosquitto installed:
net start mosquitto

# Test MQTT connection
python -c "import paho.mqtt.client as mqtt; c = mqtt.Client(); print('MQTT available:', c.connect('localhost', 1883, 60) == 0)"
```

**2. Missing Dependencies:**
```powershell
# Install missing packages
pip install paho-mqtt streamlit jsonschema pydantic voluptuous

# Check specific imports
python -c "import streamlit, paho.mqtt.client; print('All dependencies available')"
```

**3. Model File Not Found:**
```powershell
# Verify Phase 2 completed successfully
Get-ChildItem .\artifacts_phase2\ | Where-Object {$_.Name -like "*model*"}
```

**4. Port Already in Use:**
```powershell
# Check what's using port 8501 (Streamlit default)
netstat -ano | findstr :8501

# Use different port
streamlit run .\Phase_Five.py --server.port 8502
```

**5. Dashboard Not Updating:**
- Ensure Phase 3 is running and connected to MQTT
- Check MQTT topic configuration matches between phases
- Verify test data is being published

---

## Performance Notes

- **Phase 3**: Optimized for real-time inference, uses trained model from Phase 2
- **Phase 4**: Provides quantization options for edge deployment (faster inference)
- **Phase 5**: Dashboard updates in real-time, configurable refresh rates
- **System Requirements**: Recommended 4GB+ RAM, modern CPU for smooth operation

Your model achieved **99.98% accuracy** in Phase 2, so you can expect excellent anomaly detection performance in production!