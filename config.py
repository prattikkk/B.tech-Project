"""Centralized configuration using Pydantic BaseSettings.

Provides typed, validated runtime configuration. In production (ENV=prod)
we fail fast if mandatory secure settings are missing.
"""
from __future__ import annotations
from pydantic import BaseSettings, Field, validator
from typing import Optional
import os

class Settings(BaseSettings):
    # Environment / mode
    env: str = Field("dev", env="ENV")

    # MQTT
    mqtt_broker: str = Field("localhost", env="MQTT_BROKER")
    mqtt_port: int = Field(1883, env="MQTT_PORT")
    mqtt_topic: str = Field("iot/traffic", env="MQTT_TOPIC")
    mqtt_health_topic: str = Field("iot/traffic/health", env="MQTT_HEALTH_TOPIC")
    mqtt_predictions_topic: str = Field("iot/traffic/predictions", env="MQTT_PREDICTIONS_TOPIC")
    mqtt_username: Optional[str] = Field(None, env="MQTT_USERNAME")
    mqtt_password: Optional[str] = Field(None, env="MQTT_PASSWORD")
    mqtt_tls: bool = Field(False, env="MQTT_TLS")
    mqtt_tls_ca: Optional[str] = Field(None, env="MQTT_TLS_CA")
    mqtt_tls_cert: Optional[str] = Field(None, env="MQTT_TLS_CERT")
    mqtt_tls_key: Optional[str] = Field(None, env="MQTT_TLS_KEY")

    # Threshold / inference (with unified threshold management support)
    alert_threshold: float = Field(0.5, env="ALERT_THRESHOLD")
    anomaly_threshold: float = Field(0.495, env="ANOMALY_THRESHOLD")  # Phase 5 compatibility
    temperature: float = Field(1.0, env="PHASE4_FORCE_TEMPERATURE")   # Temperature scaling
    micro_batch_size: int = Field(8, env="MICRO_BATCH_SIZE")
    micro_batch_latency_ms: int = Field(20, env="MICRO_BATCH_LATENCY_MS")

    # Security / validation toggles
    validate_ingress: bool = Field(False, env="VALIDATE_INGRESS_SCHEMA")
    require_hash_match: bool = Field(False, env="REQUIRE_HASH_MATCH")

    # Metrics / performance
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(9108, env="METRICS_PORT")
    omp_threads: int = Field(1, env="OMP_NUM_THREADS")

    # Queue limits
    proc_queue_max: int = Field(5000, env="PROC_QUEUE_MAX")

    # Health / intervals
    health_interval: int = Field(15, env="HEALTH_INTERVAL")

    # Logging
    log_rotate_bytes: int = Field(10_000_000, env="LOG_ROTATE_BYTES")
    log_format: str = Field("plain", env="LOG_FORMAT")

    class Config:
        case_sensitive = False
        env_file = None

    @validator("env")
    def _normalize_env(cls, v):
        return v.lower()

    @validator("mqtt_tls", pre=True)
    def _boolify(cls, v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1","true","yes")

    @validator("validate_ingress","require_hash_match","enable_metrics")
    def _boolify_many(cls, v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1","true","yes")

    def enforce_prod_requirements(self):
        """Fail fast in production if insecure configuration is detected."""
        if self.env == "prod":
            missing = []
            if not self.mqtt_tls:
                missing.append("TLS (set MQTT_TLS=1)")
            if self.mqtt_username and not self.mqtt_password:
                missing.append("MQTT_PASSWORD for provided MQTT_USERNAME")
            if missing:
                raise RuntimeError(f"Production config invalid: {', '.join(missing)}")

_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[arg-type]
        _settings.enforce_prod_requirements()
    return _settings

if __name__ == "__main__":  # simple debug
    s = get_settings()
    for k,v in s.dict().items():
        print(f"{k}={v}")
