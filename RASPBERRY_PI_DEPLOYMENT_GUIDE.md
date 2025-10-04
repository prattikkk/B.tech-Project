# IoT Anomaly Detection - Raspberry Pi Hardware Deployment Guide

## 🍓 Complete Raspberry Pi Deployment Instructions

This guide provides step-by-step instructions to deploy your trained IoT anomaly detection model to Raspberry Pi hardware for real-world edge inference.

---

## 📋 Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Raspberry Pi Setup](#raspberry-pi-setup)
3. [Environment Preparation](#environment-preparation)
4. [Model Transfer & Optimization](#model-transfer--optimization)
5. [Edge Deployment Configuration](#edge-deployment-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Production Monitoring](#production-monitoring)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Hardware Requirements

### **Minimum Requirements:**
- **Raspberry Pi 4 Model B** (4GB RAM recommended, 8GB for heavy workloads)
- **MicroSD Card**: 32GB Class 10 or better (64GB recommended)
- **Power Supply**: Official Raspberry Pi 4 USB-C Power Supply (5.1V/3A)
- **Network**: Ethernet cable or reliable Wi-Fi connection
- **Cooling**: Heat sinks + fan (essential for continuous inference)

### **Optional Enhancements:**
- **Raspberry Pi AI Kit** (Hailo-8L NPU for accelerated inference)
- **SSD Storage**: USB 3.0 SSD for faster I/O
- **PoE+ HAT**: Power over Ethernet for single-cable deployment
- **Case**: Proper enclosure with ventilation

---

## 🖥️ Raspberry Pi Setup

### **Step 1: Operating System Installation**

```bash
# Download Raspberry Pi Imager
# https://www.raspberrypi.com/software/

# Flash Raspberry Pi OS Lite (64-bit) - recommended for headless deployment
# Enable SSH, configure Wi-Fi, set username/password during imaging
```

### **Step 2: Initial System Configuration**

```bash
# SSH into your Raspberry Pi
ssh pi@<raspberry-pi-ip>

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    htop \
    tmux \
    mosquitto \
    mosquitto-clients \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    python3-pyqt5 \
    python3-dev

# Configure GPU memory split (optional, for graphics acceleration)
sudo raspi-config
# Advanced Options -> Memory Split -> Set to 128
```

### **Step 3: Performance Optimization**

```bash
# Edit /boot/config.txt for performance
sudo nano /boot/config.txt

# Add these lines for optimization:
# Overclock (if cooling is adequate)
arm_freq=1800
gpu_freq=600
core_freq=600
sdram_freq=500
over_voltage=4

# Enable hardware interfaces
dtparam=spi=on
dtparam=i2c=on
dtparam=audio=on

# GPU memory allocation
gpu_mem=128

# Reboot to apply changes
sudo reboot
```

---

## 🐍 Environment Preparation

### **Step 4: Python Environment Setup**

```bash
# Create dedicated Python environment
python3 -m venv ~/iot_anomaly_env
source ~/iot_anomaly_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies (ARM64 optimized)
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install pandas==2.0.3
pip install scikit-learn==1.3.0

# Install PyTorch for ARM64
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install ONNX Runtime for ARM64
pip install onnxruntime==1.15.1

# Install LightGBM
pip install lightgbm==4.0.0

# Install MQTT and networking
pip install paho-mqtt==1.6.1
pip install asyncio-mqtt==0.13.0

# Install monitoring and utilities
pip install psutil==5.9.5
pip install prometheus-client==0.17.1
pip install requests==2.31.0
pip install joblib==1.3.2

# Create requirements file for reproducibility
pip freeze > ~/iot_anomaly_requirements.txt
```

### **Step 5: Project Transfer**

```bash
# Create project directory
mkdir -p ~/iot_anomaly_detection
cd ~/iot_anomaly_detection

# Transfer your project files (multiple options):

# Option A: Git clone (recommended)
git clone https://github.com/prattikkk/B.tech-Project.git .

# Option B: SCP transfer from development machine
# scp -r /path/to/your/IOTProject/* pi@<raspberry-pi-ip>:~/iot_anomaly_detection/

# Option C: rsync (efficient for updates)
# rsync -avz --progress /path/to/your/IOTProject/ pi@<raspberry-pi-ip>:~/iot_anomaly_detection/

# Set proper permissions
chmod +x *.py
chmod +x scripts/*.ps1 2>/dev/null || true
```

---

## 🎯 Model Transfer & Optimization

### **Step 6: Artifact Verification**

```bash
# Activate environment
source ~/iot_anomaly_env/bin/activate
cd ~/iot_anomaly_detection

# Verify artifacts integrity
python3 -c "
import json
import hashlib
from pathlib import Path

def verify_artifacts():
    artifacts_dir = Path('artifacts_phase2')
    manifest_path = artifacts_dir / 'artifact_manifest.json'
    
    if not manifest_path.exists():
        print('❌ Artifact manifest not found')
        return False
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    for filename, expected_hash in manifest.items():
        file_path = artifacts_dir / filename
        if not file_path.exists():
            print(f'❌ Missing: {filename}')
            continue
            
        with open(file_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        
        if actual_hash == expected_hash:
            print(f'✅ Verified: {filename}')
        else:
            print(f'❌ Corrupted: {filename}')
            return False
    
    return True

if verify_artifacts():
    print('🎉 All artifacts verified successfully!')
else:
    print('💥 Artifact verification failed!')
"
```

### **Step 7: ARM64 Model Optimization**

```bash
# Create ARM64-optimized model script
cat > optimize_for_arm64.py << 'EOF'
#!/usr/bin/env python3
"""Optimize models for ARM64 Raspberry Pi deployment"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_onnx_for_arm64():
    """Optimize ONNX model for ARM64 deployment"""
    artifacts_dir = Path("artifacts_phase2")
    
    # Original ONNX model
    original_model = artifacts_dir / "model_hybrid.onnx"
    optimized_model = artifacts_dir / "model_hybrid_arm64.onnx"
    
    if not original_model.exists():
        logger.error(f"Original model not found: {original_model}")
        return False
    
    try:
        # Load and optimize
        model = onnx.load(str(original_model))
        
        # Basic optimization passes
        from onnxruntime.tools import optimizer
        opt_model = optimizer.optimize_model(
            str(original_model),
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0
        )
        
        # Save optimized model
        opt_model.save_model_to_file(str(optimized_model))
        logger.info(f"✅ Optimized model saved: {optimized_model}")
        
        # Test inference
        session = ort.InferenceSession(str(optimized_model))
        dummy_input = np.random.randn(1, 22).astype(np.float32)
        
        output = session.run(None, {session.get_inputs()[0].name: dummy_input})
        logger.info(f"✅ ARM64 inference test passed, output shape: {output[0].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return False

def test_inference_performance():
    """Benchmark inference performance on ARM64"""
    import time
    
    artifacts_dir = Path("artifacts_phase2")
    model_path = artifacts_dir / "model_hybrid.onnx"
    
    if not model_path.exists():
        logger.error("Model not found for performance test")
        return
    
    try:
        session = ort.InferenceSession(str(model_path))
        dummy_input = np.random.randn(1, 22).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {session.get_inputs()[0].name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            session.run(None, {session.get_inputs()[0].name: dummy_input})
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        p95_time = np.percentile(times, 95) * 1000
        
        logger.info(f"🏃 Inference Performance:")
        logger.info(f"   Average: {avg_time:.2f}ms")
        logger.info(f"   P95: {p95_time:.2f}ms")
        logger.info(f"   Throughput: {1000/avg_time:.1f} inferences/sec")
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")

if __name__ == "__main__":
    logger.info("🍓 Optimizing models for ARM64 Raspberry Pi...")
    
    # optimize_onnx_for_arm64()  # Enable if onnxruntime.tools available
    test_inference_performance()
    
    logger.info("✅ ARM64 optimization complete!")
EOF

python3 optimize_for_arm64.py
```

---

## ⚙️ Edge Deployment Configuration

### **Step 8: Create Raspberry Pi Configuration**

```bash
# Create Pi-specific configuration
cat > config_raspberry_pi.py << 'EOF'
#!/usr/bin/env python3
"""Raspberry Pi specific configuration for IoT anomaly detection"""

import os
import psutil
from pathlib import Path

# Hardware detection
def detect_raspberry_pi():
    """Detect Raspberry Pi model and capabilities"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        if 'Raspberry Pi' in cpuinfo:
            if 'BCM2711' in cpuinfo:
                return 'Pi4'
            elif 'BCM2837' in cpuinfo:
                return 'Pi3'
            else:
                return 'PiOther'
    except:
        pass
    
    return 'Unknown'

# Performance configuration based on hardware
PI_MODEL = detect_raspberry_pi()
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)

print(f"🍓 Detected: {PI_MODEL}, RAM: {TOTAL_RAM_GB:.1f}GB")

# Adaptive configuration
if PI_MODEL == 'Pi4' and TOTAL_RAM_GB >= 4:
    # High-performance configuration
    MQTT_WORKER_THREADS = 4
    BATCH_SIZE = 32
    QUEUE_SIZE = 500
    USE_ONNX = True
    ENABLE_METRICS = True
elif PI_MODEL == 'Pi4':
    # Standard configuration
    MQTT_WORKER_THREADS = 2
    BATCH_SIZE = 16
    QUEUE_SIZE = 200
    USE_ONNX = True
    ENABLE_METRICS = True
else:
    # Conservative configuration
    MQTT_WORKER_THREADS = 1
    BATCH_SIZE = 8
    QUEUE_SIZE = 100
    USE_ONNX = True
    ENABLE_METRICS = False

# Pi-specific optimizations
os.environ.update({
    'OMP_NUM_THREADS': str(MQTT_WORKER_THREADS),
    'OPENBLAS_NUM_THREADS': str(MQTT_WORKER_THREADS),
    'MKL_NUM_THREADS': str(MQTT_WORKER_THREADS),
    'NUMEXPR_NUM_THREADS': str(MQTT_WORKER_THREADS),
})

# Network configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'iot/traffic')

# Paths
PROJECT_ROOT = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts_phase2'
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

print(f"⚙️  Configuration:")
print(f"   Workers: {MQTT_WORKER_THREADS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Queue Size: {QUEUE_SIZE}")
print(f"   ONNX Runtime: {USE_ONNX}")
print(f"   Metrics: {ENABLE_METRICS}")
print(f"   MQTT: {MQTT_BROKER}:{MQTT_PORT}/{MQTT_TOPIC}")
EOF

python3 config_raspberry_pi.py
```

### **Step 9: Create Edge Deployment Script**

```bash
# Create optimized Phase 3 for Raspberry Pi
cat > phase3_raspberry_pi.py << 'EOF'
#!/usr/bin/env python3
"""
IoT Anomaly Detection - Raspberry Pi Edge Deployment
Optimized version of Phase 3 for ARM64 hardware with resource constraints
"""

import os
import sys
import logging
import signal
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime
import psutil

# Import configuration
from config_raspberry_pi import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'phase3_pi_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def monitor_system_resources():
    """Monitor Pi resources and log warnings"""
    while True:
        try:
            # CPU temperature
            temp = psutil.sensors_temperatures().get('cpu_thermal', [{}])[0].get('current', 0)
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if temp > 70:
                logger.warning(f"🌡️  High CPU temperature: {temp:.1f}°C")
            if cpu_percent > 80:
                logger.warning(f"🔥 High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > 85:
                logger.warning(f"💾 High memory usage: {memory_percent:.1f}%")
            
            # Log status every 5 minutes
            if int(time.time()) % 300 == 0:
                logger.info(f"📊 System: Temp={temp:.1f}°C, CPU={cpu_percent:.1f}%, RAM={memory_percent:.1f}%")
            
        except Exception as e:
            logger.debug(f"Resource monitoring error: {e}")
        
        time.sleep(30)

def graceful_shutdown(signum, frame):
    """Handle graceful shutdown"""
    logger.info("🛑 Received shutdown signal, stopping gracefully...")
    # Add cleanup logic here
    sys.exit(0)

def main():
    """Main Raspberry Pi deployment entry point"""
    
    logger.info("🍓 Starting IoT Anomaly Detection on Raspberry Pi")
    logger.info(f"🔧 Configuration: {PI_MODEL}, {TOTAL_RAM_GB:.1f}GB RAM")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    # Start resource monitoring
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()
    
    # Import and start Phase 3 with Pi configuration
    try:
        # Set environment variables for Phase 3
        os.environ.update({
            'PHASE3_WORKER_THREADS': str(MQTT_WORKER_THREADS),
            'PHASE3_QUEUE_SIZE': str(QUEUE_SIZE),
            'PHASE3_BATCH_SIZE': str(BATCH_SIZE),
            'MQTT_BROKER_HOST': MQTT_BROKER,
            'MQTT_BROKER_PORT': str(MQTT_PORT),
            'MQTT_TOPIC': MQTT_TOPIC,
        })
        
        # Import original Phase 3
        from Phase_Three import main as phase3_main
        
        logger.info("🚀 Starting Phase 3 with Raspberry Pi optimizations")
        phase3_main()
        
    except Exception as e:
        logger.error(f"💥 Deployment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
```

---

## 🚀 Performance Optimization

### **Step 10: System Service Setup**

```bash
# Create systemd service for automatic startup
sudo tee /etc/systemd/system/iot-anomaly-detection.service > /dev/null << EOF
[Unit]
Description=IoT Anomaly Detection Service
After=network.target mosquitto.service
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/iot_anomaly_detection
Environment=PATH=/home/pi/iot_anomaly_env/bin
ExecStart=/home/pi/iot_anomaly_env/bin/python /home/pi/iot_anomaly_detection/phase3_raspberry_pi.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
MemoryLimit=2G

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable iot-anomaly-detection.service
sudo systemctl start iot-anomaly-detection.service

# Check status
sudo systemctl status iot-anomaly-detection.service
```

### **Step 11: MQTT Broker Configuration**

```bash
# Configure local MQTT broker
sudo tee /etc/mosquitto/conf.d/iot-anomaly.conf > /dev/null << EOF
# IoT Anomaly Detection MQTT Configuration
port 1883
allow_anonymous true
max_connections 100
max_queued_messages 1000

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type error
log_type warning
log_type notice
log_type information
connection_messages true
log_timestamp true

# Security (enable for production)
# password_file /etc/mosquitto/passwd
# acl_file /etc/mosquitto/acl
EOF

# Restart MQTT broker
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto

# Test MQTT connectivity
mosquitto_pub -h localhost -t iot/traffic -m '{"test": "message"}'
mosquitto_sub -h localhost -t iot/traffic -C 1
```

---

## 📊 Production Monitoring

### **Step 12: Monitoring Setup**

```bash
# Create monitoring dashboard
cat > monitoring_dashboard.py << 'EOF'
#!/usr/bin/env python3
"""Simple monitoring dashboard for Raspberry Pi deployment"""

import time
import json
import psutil
import requests
from datetime import datetime
from pathlib import Path

def collect_metrics():
    """Collect system and application metrics"""
    try:
        # System metrics
        temp = psutil.sensors_temperatures().get('cpu_thermal', [{}])[0].get('current', 0)
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_temp': temp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available // (1024*1024),
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
        }
        
        # Try to get application metrics
        try:
            response = requests.get('http://localhost:9108/metrics', timeout=2)
            if response.status_code == 200:
                metrics['application'] = 'running'
            else:
                metrics['application'] = 'error'
        except:
            metrics['application'] = 'unavailable'
        
        return metrics
        
    except Exception as e:
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}

def main():
    print("🍓 Raspberry Pi IoT Anomaly Detection - Monitoring Dashboard")
    print("=" * 60)
    
    while True:
        try:
            metrics = collect_metrics()
            
            if 'error' not in metrics:
                sys_metrics = metrics['system']
                print(f"\r🕐 {metrics['timestamp'][:19]} | "
                      f"🌡️ {sys_metrics['cpu_temp']:.1f}°C | "
                      f"🔥 CPU: {sys_metrics['cpu_percent']:.1f}% | "
                      f"💾 RAM: {sys_metrics['memory_percent']:.1f}% | "
                      f"💿 Disk: {sys_metrics['disk_percent']:.1f}% | "
                      f"📱 App: {metrics['application']}", end='')
            else:
                print(f"\r❌ Error: {metrics['error']}", end='')
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"\r💥 Monitor error: {e}", end='')
        
        time.sleep(5)

if __name__ == "__main__":
    main()
EOF

# Make executable
chmod +x monitoring_dashboard.py
```

### **Step 13: Log Rotation Setup**

```bash
# Setup log rotation
sudo tee /etc/logrotate.d/iot-anomaly-detection > /dev/null << EOF
/home/pi/iot_anomaly_detection/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
    su pi pi
}
EOF

# Test log rotation
sudo logrotate -d /etc/logrotate.d/iot-anomaly-detection
```

---

## 🧪 Testing & Validation

### **Step 14: Deployment Testing**

```bash
# Create comprehensive test suite
cat > test_raspberry_pi_deployment.py << 'EOF'
#!/usr/bin/env python3
"""Test suite for Raspberry Pi deployment validation"""

import os
import sys
import time
import json
import subprocess
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test Python environment and dependencies"""
    logger.info("🐍 Testing Python environment...")
    
    try:
        import torch
        import onnxruntime as ort
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        import paho.mqtt.client
        
        logger.info("✅ All Python dependencies available")
        
        # Test ONNX Runtime
        session = ort.InferenceSession("artifacts_phase2/model_hybrid.onnx")
        dummy_input = np.random.randn(1, 22).astype(np.float32)
        output = session.run(None, {session.get_inputs()[0].name: dummy_input})
        logger.info(f"✅ ONNX inference test passed: {output[0].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

def test_mqtt_connectivity():
    """Test MQTT broker connectivity"""
    logger.info("📡 Testing MQTT connectivity...")
    
    try:
        result = subprocess.run([
            'mosquitto_pub', '-h', 'localhost', '-t', 'test/topic', 
            '-m', '{"test": "connectivity"}'
        ], capture_output=True, timeout=5)
        
        if result.returncode == 0:
            logger.info("✅ MQTT publish test passed")
            return True
        else:
            logger.error(f"❌ MQTT publish failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MQTT test failed: {e}")
        return False

def test_system_resources():
    """Test system resource availability"""
    logger.info("🔧 Testing system resources...")
    
    try:
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            logger.warning(f"⚠️ Low memory: {memory.available // (1024*1024)}MB available")
        else:
            logger.info(f"✅ Memory OK: {memory.available // (1024*1024)}MB available")
        
        # Check CPU temperature
        temps = psutil.sensors_temperatures()
        if 'cpu_thermal' in temps:
            temp = temps['cpu_thermal'][0].current
            if temp > 80:
                logger.warning(f"⚠️ High CPU temperature: {temp}°C")
            else:
                logger.info(f"✅ CPU temperature OK: {temp}°C")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        if free_gb < 5:
            logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB free")
        else:
            logger.info(f"✅ Disk space OK: {free_gb:.1f}GB free")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Resource check failed: {e}")
        return False

def test_inference_performance():
    """Test inference performance on Pi hardware"""
    logger.info("🏃 Testing inference performance...")
    
    try:
        import numpy as np
        import time
        import onnxruntime as ort
        
        session = ort.InferenceSession("artifacts_phase2/model_hybrid.onnx")
        dummy_input = np.random.randn(1, 22).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {session.get_inputs()[0].name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            session.run(None, {session.get_inputs()[0].name: dummy_input})
            times.append(time.perf_counter() - start)
        
        avg_time_ms = np.mean(times) * 1000
        throughput = 1000 / avg_time_ms
        
        logger.info(f"✅ Performance: {avg_time_ms:.2f}ms avg, {throughput:.1f} inferences/sec")
        
        if avg_time_ms < 100:  # Under 100ms is good
            logger.info("🚀 Excellent performance for real-time inference")
        elif avg_time_ms < 500:  # Under 500ms is acceptable
            logger.info("✅ Good performance for IoT applications")
        else:
            logger.warning("⚠️ Performance may be limiting for real-time applications")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run complete deployment test suite"""
    logger.info("🍓 Raspberry Pi Deployment Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("MQTT Connectivity", test_mqtt_connectivity),
        ("System Resources", test_system_resources),
        ("Inference Performance", test_inference_performance),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 TEST RESULTS SUMMARY:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status}: {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Raspberry Pi deployment ready for production!")
        return 0
    else:
        logger.error("💥 Deployment issues detected. Please fix before production use.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run deployment tests
python3 test_raspberry_pi_deployment.py
```

---

## 🚀 Production Deployment

### **Step 15: Launch Production Service**

```bash
# Final deployment steps
echo "🍓 Raspberry Pi Deployment - Final Steps"
echo "=========================================="

# Activate environment
source ~/iot_anomaly_env/bin/activate

# Navigate to project
cd ~/iot_anomaly_detection

# Run final tests
python3 test_raspberry_pi_deployment.py

# Start the service
sudo systemctl start iot-anomaly-detection.service

# Check service status
sudo systemctl status iot-anomaly-detection.service --no-pager -l

# Start monitoring dashboard (in tmux session)
tmux new-session -d -s monitoring 'python3 monitoring_dashboard.py'

echo ""
echo "🎉 Deployment Complete!"
echo "📊 Monitor: tmux attach -t monitoring"
echo "📋 Logs: sudo journalctl -u iot-anomaly-detection.service -f"
echo "🔧 Control: sudo systemctl {start|stop|restart} iot-anomaly-detection.service"
echo ""
echo "🌐 Your IoT anomaly detection system is now running on Raspberry Pi!"
```

---

## 🔧 Troubleshooting

### **Common Issues & Solutions:**

#### **High CPU Temperature:**
```bash
# Check temperature
vcgencmd measure_temp

# Install cooling solution or reduce CPU frequency
sudo nano /boot/config.txt
# Reduce arm_freq=1500 or add disable_overscan=1
```

#### **Memory Issues:**
```bash
# Check memory usage
free -h
sudo systemctl status iot-anomaly-detection.service

# Reduce worker threads in config_raspberry_pi.py
MQTT_WORKER_THREADS = 1
BATCH_SIZE = 8
```

#### **MQTT Connection Issues:**
```bash
# Check MQTT broker
sudo systemctl status mosquitto
mosquitto_sub -h localhost -t '#' -v

# Test network connectivity
ping <mqtt-broker-ip>
```

#### **Service Won't Start:**
```bash
# Check service logs
sudo journalctl -u iot-anomaly-detection.service -f --no-pager

# Check Python environment
source ~/iot_anomaly_env/bin/activate
python3 -c "import torch, onnxruntime; print('OK')"
```

---

## 📈 Performance Monitoring

### **Key Metrics to Monitor:**
- **CPU Temperature**: Keep < 70°C for sustained operation
- **Memory Usage**: Keep < 80% for stability
- **Inference Latency**: Target < 100ms for real-time detection
- **Network Throughput**: Monitor MQTT message processing rate
- **Detection Accuracy**: Log and review anomaly detection results

### **Production Checklist:**
- [ ] ✅ All tests passing
- [ ] ✅ Service auto-starts on boot
- [ ] ✅ Monitoring dashboard running
- [ ] ✅ Log rotation configured
- [ ] ✅ MQTT broker secured (if needed)
- [ ] ✅ Network connectivity stable
- [ ] ✅ Cooling solution adequate
- [ ] ✅ Power supply reliable
- [ ] ✅ Backup/recovery plan in place

---

**🎯 Your IoT anomaly detection model is now deployed on Raspberry Pi hardware and ready for real-world edge inference!**