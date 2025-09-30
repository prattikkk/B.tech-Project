#!/usr/bin/env python3
"""
Lightweight ONNX Ensemble Service for reduced memory footprint.

This module provides both in-process ONNX ensemble loading and optional
microservice isolation for heavy ensemble models.
"""
import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

# Optional imports with graceful fallback
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import joblib
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

logger = logging.getLogger(__name__)

class ONNXEnsembleLoader:
    """Lightweight ONNX-based ensemble loader with memory optimization."""
    
    def __init__(self, artifacts_dir: Union[str, Path]):
        self.artifacts_dir = Path(artifacts_dir)
        self.onnx_session = None
        self.input_name = None
        self.fallback_model = None
        self.ensemble_config = {}
        self._lock = threading.Lock()
        
    def load_ensemble(self, prefer_onnx: bool = True) -> bool:
        """Load ensemble model, preferring ONNX for memory efficiency.
        
        Args:
            prefer_onnx: If True, try ONNX first, fallback to pickle
            
        Returns:
            bool: True if any model loaded successfully
        """
        with self._lock:
            # Try ONNX first if preferred and available
            if prefer_onnx and ORT_AVAILABLE:
                onnx_path = self.artifacts_dir / "lgbm_model.onnx"
                if onnx_path.exists():
                    try:
                        self.onnx_session = ort.InferenceSession(
                            str(onnx_path), 
                            providers=['CPUExecutionProvider']
                        )
                        self.input_name = self.onnx_session.get_inputs()[0].name
                        logger.info("✅ Loaded ONNX ensemble from %s", onnx_path)
                        return True
                    except Exception as e:
                        logger.warning("Failed to load ONNX ensemble: %s", e)
            
            # Fallback to pickle models
            if PICKLE_AVAILABLE:
                pickle_candidates = [
                    self.artifacts_dir / "lgbm_model.pkl",
                    self.artifacts_dir / "lgbm_model.joblib",
                    self.artifacts_dir / "lightgbm_model.pkl",
                ]
                
                for path in pickle_candidates:
                    if path.exists():
                        try:
                            if path.suffix == ".joblib":
                                self.fallback_model = joblib.load(path)
                            else:
                                with open(path, 'rb') as f:
                                    self.fallback_model = pickle.load(f)
                            logger.info("✅ Loaded pickle ensemble from %s", path)
                            return True
                        except Exception as e:
                            logger.warning("Failed to load pickle ensemble from %s: %s", path, e)
            
            logger.warning("❌ No ensemble model could be loaded")
            return False
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions with automatic fallback.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Probability predictions for class 1, shape (n_samples,)
        """
        with self._lock:
            # Try ONNX first
            if self.onnx_session is not None:
                try:
                    X_input = X.astype(np.float32)
                    onnx_output = self.onnx_session.run(None, {self.input_name: X_input})[0]
                    
                    # Handle different output formats
                    if onnx_output.shape[1] == 2:  # Binary classification
                        return onnx_output[:, 1]
                    else:  # Single probability column
                        return onnx_output.flatten()
                        
                except Exception as e:
                    logger.warning("ONNX ensemble prediction failed, falling back to pickle: %s", e)
            
            # Fallback to pickle model
            if self.fallback_model is not None:
                try:
                    return self.fallback_model.predict_proba(X)[:, 1]
                except Exception as e:
                    logger.error("Fallback ensemble prediction failed: %s", e)
                    return np.zeros(X.shape[0], dtype=np.float32)
            
            # No model available
            logger.warning("No ensemble model available for prediction")
            return np.zeros(X.shape[0], dtype=np.float32)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        info = {
            "using_onnx": self.onnx_session is not None,
            "using_fallback": self.fallback_model is not None,
            "model_type": None
        }
        
        if self.onnx_session is not None:
            info["model_type"] = "onnx"
            # ONNX sessions don't expose memory info easily
            info["estimated_memory_mb"] = "low (ONNX runtime)"
        elif self.fallback_model is not None:
            info["model_type"] = "pickle"
            info["estimated_memory_mb"] = "higher (full Python object)"
        else:
            info["model_type"] = "none"
            info["estimated_memory_mb"] = 0
            
        return info


class EnsembleMicroservice:
    """Optional microservice wrapper for process isolation."""
    
    def __init__(self, port: int = 8765, artifacts_dir: Union[str, Path] = None):
        self.port = port
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path("artifacts_phase2")
        self.ensemble_loader = None
        
    def start_service(self):
        """Start lightweight HTTP service for ensemble predictions."""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json
            
            # Initialize ensemble loader
            self.ensemble_loader = ONNXEnsembleLoader(self.artifacts_dir)
            if not self.ensemble_loader.load_ensemble():
                logger.error("Failed to load ensemble for microservice")
                return False
            
            class EnsembleHandler(BaseHTTPRequestHandler):
                def do_POST(self):
                    if self.path == '/predict':
                        try:
                            content_length = int(self.headers['Content-Length'])
                            post_data = self.rfile.read(content_length)
                            data = json.loads(post_data.decode('utf-8'))
                            
                            features = np.array(data['features'], dtype=np.float32)
                            if features.ndim == 1:
                                features = features.reshape(1, -1)
                            
                            predictions = self.server.ensemble_loader.predict_proba(features)
                            
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({
                                'predictions': predictions.tolist(),
                                'status': 'success'
                            }).encode('utf-8'))
                            
                        except Exception as e:
                            self.send_response(500)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({
                                'error': str(e),
                                'status': 'error'
                            }).encode('utf-8'))
                    else:
                        self.send_response(404)
                        self.end_headers()
                        
                def do_GET(self):
                    if self.path == '/health':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        memory_info = self.server.ensemble_loader.get_memory_info()
                        self.wfile.write(json.dumps({
                            'status': 'healthy',
                            'memory_info': memory_info
                        }).encode('utf-8'))
                    else:
                        self.send_response(404)
                        self.end_headers()
                        
                def log_message(self, format, *args):
                    pass  # Suppress request logging
            
            server = HTTPServer(('localhost', self.port), EnsembleHandler)
            server.ensemble_loader = self.ensemble_loader
            
            logger.info("🚀 Ensemble microservice started on port %d", self.port)
            server.serve_forever()
            
        except ImportError:
            logger.error("HTTP server not available for microservice")
            return False
        except Exception as e:
            logger.error("Failed to start ensemble microservice: %s", e)
            return False


class RemoteEnsembleClient:
    """Client for remote ensemble microservice."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.base_url = f"http://{host}:{port}"
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from remote ensemble service."""
        try:
            import urllib.request
            import urllib.parse
            
            data = json.dumps({'features': X.tolist()}).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/predict",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=5.0) as response:
                result = json.loads(response.read().decode('utf-8'))
                
            if result['status'] == 'success':
                return np.array(result['predictions'], dtype=np.float32)
            else:
                logger.error("Remote ensemble prediction error: %s", result.get('error'))
                return np.zeros(X.shape[0], dtype=np.float32)
                
        except Exception as e:
            logger.warning("Remote ensemble call failed: %s", e)
            return np.zeros(X.shape[0], dtype=np.float32)
    
    def health_check(self) -> bool:
        """Check if remote ensemble service is healthy."""
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('status') == 'healthy'
        except Exception:
            return False


# Convenience function for easy integration
def create_ensemble_loader(artifacts_dir: Union[str, Path], 
                          prefer_onnx: bool = True,
                          use_microservice: bool = False,
                          microservice_port: int = 8765) -> Union[ONNXEnsembleLoader, RemoteEnsembleClient]:
    """Create appropriate ensemble loader based on configuration.
    
    Args:
        artifacts_dir: Directory containing ensemble models
        prefer_onnx: Whether to prefer ONNX over pickle models
        use_microservice: Whether to use process isolation via microservice
        microservice_port: Port for microservice if enabled
        
    Returns:
        Ensemble loader instance (local or remote)
    """
    if use_microservice:
        # Check if microservice is already running
        client = RemoteEnsembleClient(port=microservice_port)
        if client.health_check():
            logger.info("✅ Using existing ensemble microservice on port %d", microservice_port)
            return client
        else:
            logger.warning("⚠️ Microservice requested but not available, using local loader")
    
    # Use local ONNX loader
    loader = ONNXEnsembleLoader(artifacts_dir)
    if loader.load_ensemble(prefer_onnx=prefer_onnx):
        return loader
    else:
        logger.error("❌ Failed to create ensemble loader")
        return None


if __name__ == "__main__":
    # CLI for starting microservice
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble Microservice")
    parser.add_argument("--port", type=int, default=8765, help="Service port")
    parser.add_argument("--artifacts", type=str, default="artifacts_phase2", 
                       help="Artifacts directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    service = EnsembleMicroservice(args.port, args.artifacts)
    service.start_service()