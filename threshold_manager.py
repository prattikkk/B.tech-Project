#!/usr/bin/env python3
"""
Unified Threshold Management for IoT Anomaly Detection Pipeline

Provides centralized threshold resolution with proper precedence:
1. CLI arguments (highest priority)  
2. Environment variables
3. evaluation.json from Phase 2 (canonical source)
4. Default fallback values (lowest priority)

This ensures consistent threshold values across all phases and components.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class ThresholdManager:
    """Centralized threshold management with proper precedence resolution."""
    
    # Default fallback values
    DEFAULT_VALUES = {
        'threshold': 0.5,
        'temperature': 1.0,
        'alert_threshold': 0.5,  # alias for threshold
        'anomaly_threshold': 0.495,  # Phase 5 specific
        'active_threshold': None,  # computed based on ensemble mode
    }
    
    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None):
        """
        Initialize threshold manager.
        
        Args:
            artifacts_dir: Path to artifacts directory (defaults to artifacts_phase2)
        """
        if artifacts_dir is None:
            base_dir = Path(__file__).resolve().parent
            artifacts_dir = base_dir / "artifacts_phase2"
        
        self.artifacts_dir = Path(artifacts_dir)
        self.evaluation_json_path = self.artifacts_dir / "evaluation.json"
        
        # Load evaluation.json if available
        self._evaluation_data = self._load_evaluation_json()
        
    def _load_evaluation_json(self) -> Dict[str, Any]:
        """Load evaluation.json with error handling."""
        if not self.evaluation_json_path.exists():
            logger.debug("evaluation.json not found at %s", self.evaluation_json_path)
            return {}
            
        try:
            with open(self.evaluation_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug("Loaded evaluation.json from %s", self.evaluation_json_path)
            return data
        except Exception as e:
            logger.warning("Failed to load evaluation.json: %s", e)
            return {}
    
    def get_threshold(self, 
                     cli_value: Optional[float] = None,
                     env_var: str = "ALERT_THRESHOLD",
                     ensemble_enabled: bool = False) -> float:
        """
        Get threshold with proper precedence resolution.
        
        Args:
            cli_value: CLI argument value (highest priority)
            env_var: Environment variable name to check
            ensemble_enabled: Whether ensemble mode is enabled
            
        Returns:
            Resolved threshold value
        """
        # 1. CLI argument (highest priority)
        if cli_value is not None:
            logger.debug("Using CLI threshold: %.4f", cli_value)
            return float(cli_value)
        
        # 2. Environment variable
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                threshold = float(env_value)
                logger.debug("Using environment threshold (%s): %.4f", env_var, threshold)
                return threshold
            except ValueError:
                logger.warning("Invalid environment threshold value %s=%s, ignoring", env_var, env_value)
        
        # 3. evaluation.json (canonical source from Phase 2)
        if self._evaluation_data:
            # Choose appropriate threshold based on ensemble mode
            if ensemble_enabled:
                # Use full ensemble threshold
                eval_threshold = self._evaluation_data.get("meta", {}).get("val_best_threshold")
            else:
                # Use deep-only threshold if available, otherwise fall back to full threshold
                eval_threshold = (
                    self._evaluation_data.get("meta", {}).get("val_best_threshold_deep_only") or
                    self._evaluation_data.get("meta", {}).get("val_best_threshold")
                )
            
            if eval_threshold is not None:
                threshold = float(eval_threshold)
                logger.debug("Using evaluation.json threshold (ensemble=%s): %.4f", ensemble_enabled, threshold)
                return threshold
        
        # 4. Default fallback
        default_threshold = self.DEFAULT_VALUES['threshold']
        logger.debug("Using default threshold: %.4f", default_threshold)
        return default_threshold
    
    def get_temperature(self,
                       cli_value: Optional[float] = None,
                       env_var: str = "PHASE4_FORCE_TEMPERATURE") -> float:
        """
        Get temperature with proper precedence resolution.
        
        Args:
            cli_value: CLI argument value (highest priority)  
            env_var: Environment variable name to check
            
        Returns:
            Resolved temperature value
        """
        # 1. CLI argument (highest priority)
        if cli_value is not None:
            logger.debug("Using CLI temperature: %.4f", cli_value)
            return float(cli_value)
        
        # 2. Environment variable
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                temperature = float(env_value)
                logger.debug("Using environment temperature (%s): %.4f", env_var, temperature)
                return temperature
            except ValueError:
                logger.warning("Invalid environment temperature value %s=%s, ignoring", env_var, env_value)
        
        # 3. evaluation.json (canonical source from Phase 2)
        if self._evaluation_data:
            eval_temperature = self._evaluation_data.get("calibration", {}).get("temperature")
            if eval_temperature is not None:
                temperature = float(eval_temperature)
                # Clamp sub-unity temperatures (amplify logits, likely mis-saved)
                if temperature < 1.0:
                    logger.warning("Clamping temperature %.4f -> 1.0 (sub-unity temperatures amplify logits)", temperature)
                    temperature = 1.0
                logger.debug("Using evaluation.json temperature: %.4f", temperature)
                return temperature
        
        # 4. Default fallback
        default_temperature = self.DEFAULT_VALUES['temperature']
        logger.debug("Using default temperature: %.4f", default_temperature)
        return default_temperature
    
    def get_all_thresholds(self,
                          cli_threshold: Optional[float] = None,
                          cli_temperature: Optional[float] = None,
                          ensemble_enabled: bool = False) -> Dict[str, float]:
        """
        Get all threshold values with consistent precedence.
        
        Args:
            cli_threshold: CLI threshold argument
            cli_temperature: CLI temperature argument
            ensemble_enabled: Whether ensemble mode is enabled
            
        Returns:
            Dictionary of resolved threshold values
        """
        threshold = self.get_threshold(cli_threshold, ensemble_enabled=ensemble_enabled)
        temperature = self.get_temperature(cli_temperature)
        
        return {
            'threshold': threshold,
            'temperature': temperature,
            'active_threshold': threshold,  # Same as threshold for now
            'alert_threshold': threshold,   # Alias
            'anomaly_threshold': threshold, # Phase 5 compatibility
        }
    
    def log_threshold_summary(self, values: Dict[str, float]):
        """Log a summary of resolved threshold values."""
        logger.info("Threshold Summary: threshold=%.4f temperature=%.4f active_threshold=%.4f",
                   values['threshold'], values['temperature'], values['active_threshold'])
        
        # Add source information if available
        sources = []
        if self._evaluation_data:
            sources.append("evaluation.json")
        if any(os.getenv(var) for var in ["ALERT_THRESHOLD", "PHASE4_FORCE_TEMPERATURE"]):
            sources.append("environment")
        if not sources:
            sources.append("defaults")
            
        logger.info("Threshold sources: %s", ", ".join(sources))

# Global instance for easy access
_global_manager = None

def get_threshold_manager(artifacts_dir: Optional[Union[str, Path]] = None) -> ThresholdManager:
    """Get global threshold manager instance."""
    global _global_manager
    if _global_manager is None or artifacts_dir is not None:
        _global_manager = ThresholdManager(artifacts_dir)
    return _global_manager

def get_unified_thresholds(cli_threshold: Optional[float] = None,
                          cli_temperature: Optional[float] = None,
                          ensemble_enabled: bool = False,
                          artifacts_dir: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """
    Convenience function to get unified thresholds.
    
    Args:
        cli_threshold: CLI threshold argument
        cli_temperature: CLI temperature argument  
        ensemble_enabled: Whether ensemble mode is enabled
        artifacts_dir: Path to artifacts directory
        
    Returns:
        Dictionary of resolved threshold values
    """
    manager = get_threshold_manager(artifacts_dir)
    values = manager.get_all_thresholds(cli_threshold, cli_temperature, ensemble_enabled)
    manager.log_threshold_summary(values)
    return values

if __name__ == "__main__":
    # Example usage and test
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Threshold Manager Test ===")
    
    # Test with defaults
    print("\n1. Default values:")
    values = get_unified_thresholds()
    for key, value in values.items():
        print(f"  {key}: {value}")
    
    # Test with CLI overrides
    print("\n2. With CLI overrides:")
    values = get_unified_thresholds(cli_threshold=0.7, cli_temperature=1.5)
    for key, value in values.items():
        print(f"  {key}: {value}")
    
    # Test with ensemble mode
    print("\n3. With ensemble enabled:")
    values = get_unified_thresholds(ensemble_enabled=True)
    for key, value in values.items():
        print(f"  {key}: {value}")