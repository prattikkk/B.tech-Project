"""Reusable rotating CSV logger with disk safeguards.

Features:
- Asynchronous queue-based writer (non-blocking produce)
- Size-based rotation per file (max_bytes, backup_count)
- Global cumulative size cap across primary + rotated files (total_cap_bytes)
- Disk free space watermark skip logic (min_free_disk_mb)
- Metrics hooks: increments provided metrics dict keys
- Graceful shutdown

Environment variable overrides (fallback to constructor):
  LOG_MAX_BYTES: per-file max size before rotation
  LOG_BACKUP_COUNT: number of rotated files to retain (N oldest pruned first if over)
  LOG_MIN_FREE_DISK_MB / MIN_FREE_DISK_MB: minimum free space required to write
  LOG_TOTAL_CAP_BYTES: global cumulative cap across primary+rotated (older pruned)

Usage:
  logger = RotatingCSVLogger(Path('phase3_predictions_log.csv'), metrics=_metrics, metrics_lock=_metrics_lock)
  logger.log(ts, true_label, pred, prob)
  logger.close()
"""
from __future__ import annotations
from pathlib import Path
import threading, queue, time, os, shutil
from typing import Iterable, Optional, List, Dict, Any, Callable

class RotatingCSVLogger:
    def __init__(self,
                 path: Path,
                 header: Optional[Iterable[str]] = None,
                 max_queue: int = 20000,
                 flush_interval: float = 0.5,
                 max_bytes: Optional[int] = None,
                 backup_count: int = 5,
                 min_free_disk_mb: int = 50,
                 total_cap_bytes: Optional[int] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 metrics_lock: Optional[threading.Lock] = None,
                 disk_free_func: Optional[Callable[[Path], int]] = None):
        self.path = Path(path)
        self.header = list(header) if header else ["timestamp","true_label","pred","prob_attack"]
        self.queue: "queue.Queue[tuple]" = queue.Queue(maxsize=max_queue)
        self.flush_interval = flush_interval
        # Explicit args take precedence over environment variables
        self.max_bytes = int(str(max_bytes)) if max_bytes is not None else int(os.getenv("LOG_MAX_BYTES", "10000000"))
        self.backup_count = int(str(backup_count)) if backup_count is not None else int(os.getenv("LOG_BACKUP_COUNT", "5"))
        self.min_free_disk_mb = int(str(min_free_disk_mb)) if min_free_disk_mb is not None else int(os.getenv("LOG_MIN_FREE_DISK_MB", os.getenv("MIN_FREE_DISK_MB", "50")))
        self.total_cap_bytes = int(str(total_cap_bytes)) if total_cap_bytes is not None else (int(os.getenv("LOG_TOTAL_CAP_BYTES", "0")) or None)
        self.metrics = metrics
        self.metrics_lock = metrics_lock
        self.disk_free_func = disk_free_func or self._default_disk_free
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._ensure_file()
        self._thread.start()

    # -------------- Public API --------------
    def log(self, timestamp, true_label, pred, prob_attack):
        try:
            self.queue.put_nowait((timestamp, true_label, pred, prob_attack))
        except queue.Full:
            # drop silently (could track another metric if desired)
            pass

    def close(self, timeout: float = 2.0):
        self._stop_event.set()
        self._thread.join(timeout=timeout)
    # Backward compatibility with previous AsyncCSVLogger.stop()
    def stop(self, timeout: float = 2.0):  # pragma: no cover - simple alias
        self.close(timeout=timeout)

    # -------------- Internal helpers --------------
    def _default_disk_free(self, path: Path) -> int:
        try:
            usage = shutil.disk_usage(str(path.parent))
            return int(usage.free // (1024 * 1024))
        except Exception:
            return 999999  # treat as plenty free

    def _ensure_file(self):
        if not self.path.parent.exists():
            try: self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception: return
        if not self.path.exists():
            try:
                with self.path.open('w', encoding='utf-8') as f:
                    f.write(",".join(self.header) + "\n")
            except Exception:
                pass

    def _maybe_rotate(self):
        try:
            if not self.path.exists():
                return
            if self.path.stat().st_size < self.max_bytes:
                return
            # Rotate: shift backups
            for i in range(self.backup_count, 0, -1):
                src = self.path.with_suffix(self.path.suffix + f".{i}") if i > 0 else self.path
                dst = self.path.with_suffix(self.path.suffix + f".{i+1}")
                if i == self.backup_count:
                    # remove oldest if exists
                    if dst.exists():
                        try: dst.unlink()
                        except Exception: pass
                if src.exists():
                    try: src.rename(dst)
                    except Exception: pass
            # Rename current to .1, recreate base
            rotated = self.path.with_suffix(self.path.suffix + ".1")
            try:
                self.path.rename(rotated)
            except Exception:
                return
            try:
                with self.path.open('w', encoding='utf-8') as f:
                    f.write(",".join(self.header) + "\n")
            except Exception:
                pass
            self._metric_inc('log_rotations_total')
        finally:
            # After rotation enforce total cap
            self._enforce_total_cap()

    def _enforce_total_cap(self):
        if not self.total_cap_bytes:
            return
        # Collect files
        files: List[Path] = [self.path]
        # Include a generous range of rotated suffixes to guarantee pruning (even if earlier runs created more)
        for i in range(1, max(self.backup_count+2, 12)):
            p = self.path.with_suffix(self.path.suffix + f".{i}")
            if p.exists():
                files.append(p)
        # Sort oldest first by mtime
        files_sorted = sorted(files, key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files_sorted if p.exists())
        pruned_bytes = 0
        while total > self.total_cap_bytes and files_sorted:
            victim = files_sorted.pop(0)
            if victim == self.path:  # never delete active file
                break
            try:
                sz = victim.stat().st_size
                victim.unlink()
                pruned_bytes += sz
                total -= sz
            except Exception:
                break
        if pruned_bytes:
            self._metric_add('log_pruned_bytes_total', pruned_bytes)

    def _metric_inc(self, key: str, amt: int = 1):
        if self.metrics is None or self.metrics_lock is None:
            return
        with self.metrics_lock:
            self.metrics[key] = self.metrics.get(key, 0) + amt

    def _metric_add(self, key: str, amt: int):
        if self.metrics is None or self.metrics_lock is None:
            return
        with self.metrics_lock:
            self.metrics[key] = self.metrics.get(key, 0) + amt

    def _writer(self):
        buffer = []
        last_flush = time.time()
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=self.flush_interval)
                buffer.append(item)
            except queue.Empty:
                pass
            now = time.time()
            if (now - last_flush) >= self.flush_interval or len(buffer) >= 1000 or (self._stop_event.is_set() and buffer):
                self._flush(buffer)
                buffer.clear()
                last_flush = now
        # final flush
        if buffer:
            self._flush(buffer)

    def _flush(self, buffer: List[tuple]):
        if not buffer:
            return
        # Disk watermark check
        free_mb = self.disk_free_func(self.path)
        if free_mb < self.min_free_disk_mb:
            self._metric_inc('log_drops_low_disk_total', amt=len(buffer))
            return
        # Write
        try:
            with self.path.open('a', encoding='utf-8') as f:
                for row in buffer:
                    ts, tl, pred, prob = row
                    f.write(f"{ts},{tl},{pred},{prob}\n")
        except Exception:
            return
        # Rotation check
        self._maybe_rotate()
        # Enforce global total cap even if rotation not triggered (file may grow slowly below per-file threshold but cumulative chain already large)
        self._enforce_total_cap()

# Backwards compatibility alias name
AsyncCSVLogger = RotatingCSVLogger

__all__ = ["RotatingCSVLogger", "AsyncCSVLogger"]
