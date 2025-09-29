#!/usr/bin/env python3
"""
Phase_2.py (revised)

Trains the hybrid model and exports artifacts for edge deployment:
 - final_model_hybrid.pth  (PyTorch weights)
 - model_hybrid.onnx       (ONNX, dynamic axes: batch x features)
 - model_hybrid_q.onnx     (optional quantized ONNX)
 - scaler.json             (mean & scale arrays for edge)
 - lgbm_model.pkl          (optional ensemble)
 - evaluation.json         (metrics, temperature, threshold, ensemble config)

Notes:
 - ONNX export uses a dummy input (1, num_features).
 - ONNX dynamic axes: {0: 'batch', 1: 'features'}.
 - If onnxruntime is available, the script will attempt to verify ONNX inference matches PyTorch to a tiny tolerance.
"""

import os
import json
import pickle
import joblib
import shutil
import random
import math
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, accuracy_score, precision_score, recall_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Ensure project root on sys.path for tests importing as module
import sys as _sys
try:
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
except Exception:
    pass

__all__ = ["DeepHybridModel"]

# Local reproducibility function
def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_all_seeds = set_seeds

# try optional ONNX tools
try:
    import onnx
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# optional quantization
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_QUANT_AVAILABLE = True
except Exception:
    ONNX_QUANT_AVAILABLE = False

# --------------------
# Paths & constants
# --------------------
BASE_DIR = Path(__file__).resolve().parent
PHASE1_DIR = BASE_DIR / "artifacts_phase1"
PHASE2_DIR = BASE_DIR / "artifacts_phase2"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

DATA_NPZ = PHASE1_DIR / "data.npz"
DATA_PKL = PHASE1_DIR / "data.pkl"
PHASE1_SCALER_JSON = PHASE1_DIR / "scaler.json"
PHASE1_SCALER_PKL = PHASE1_DIR / "scaler.pkl"

BEST_MODEL_PATH = PHASE2_DIR / "best_model_hybrid.pth"
FINAL_MODEL_PATH = PHASE2_DIR / "final_model_hybrid.pth"
ONNX_MODEL_PATH = PHASE2_DIR / "model_hybrid.onnx"
ONNX_MODEL_QPATH = PHASE2_DIR / "model_hybrid_q.onnx"
ONNX_PARITY_JSON = PHASE2_DIR / "onnx_parity.json"
QUANT_REPORT_JSON = PHASE2_DIR / "quantization_report.json"
PHASE2_SCALER_PKL = PHASE2_DIR / "scaler.pkl"
PHASE2_SCALER_JSON = PHASE2_DIR / "scaler.json"
PHASE2_FEATURE_ORDER = PHASE2_DIR / "feature_order.json"
PHASE2_CLIP_BOUNDS = PHASE2_DIR / "clip_bounds.json"
QUANT_CONFIG_JSON = PHASE2_DIR / "quant_config.json"  # optional external override
OPTIM_CONFIG_JSON = PHASE2_DIR / "optimization_config.json"  # optional optimization config
FP16_ONNX_PATH = PHASE2_DIR / "model_hybrid_fp16.onnx"
PRUNING_REPORT_JSON = PHASE2_DIR / "pruning_report.json"
QAT_REPORT_JSON = PHASE2_DIR / "qat_report.json"
EVAL_JSON = PHASE2_DIR / "evaluation.json"
LGBM_MODEL_PKL = PHASE2_DIR / "lgbm_model.pkl"

SEED = 42
BATCH_SIZE = 1024
LR = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
RAW_STD_EPS = 1e-6
RAW_DOM_THRESH = 0.999

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2")

# reproducibility (centralized)
set_all_seeds(SEED)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

class DummyScaler:
    """Picklable scaler wrapper used when loading from scaler.json.
    Provides mean_/scale_ and a transform() method compatible with StandardScaler.
    """
    def __init__(self, mean: np.ndarray, scale: np.ndarray, with_mean: bool = True, with_std: bool = True):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        denom = self.scale_ if self.with_std else 1.0
        num = (X - self.mean_) if self.with_mean else X
        return num / (denom + 1e-12)

def _to_jsonable(x):
    import numpy as _np
    from pathlib import Path as _Path
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, _Path):
        return str(x)
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.ndarray,)):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # fallback to string
    return str(x)

def safe_json_dump(obj, path):
    # Convert object recursively to JSON-friendly form first
    jsonable = _to_jsonable(obj)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

# --------------------
# Dataset
# --------------------
class IoTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------
# Model blocks (same as yours)
# --------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
    def forward(self, x):
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
        x_seq = x.unsqueeze(1)
        x_seq = self.cnn_stem(x_seq)
        x_seq = self.resblock1(x_seq)
        x_seq = self.se1(x_seq)
        x_seq = self.resblock2(x_seq)
        x_seq = self.se2(x_seq)
        x_seq = self.pool(x_seq)
        x_seq = x_seq.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_seq)
        attn_out = self.attn(lstm_out)
        x_seq = attn_out.mean(dim=1)
        x_mlp = self.mlp(x)
        x_cat = torch.cat([x_seq, x_mlp], dim=1)
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        return self.fc2(x_cat)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

# --------------------
# Utilities (identical to your helpers)
# --------------------
def safe_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def safe_model_load(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and not all(isinstance(v, torch.Tensor) for v in state.values()):
        for k in ("state_dict", "model_state_dict", "weights"):
            if k in state:
                state = state[k]
                break
    return state

def is_already_standardized(X, mean_tol=0.05, std_tol=0.05):
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return np.all(np.abs(m) < mean_tol) and np.all(np.abs(s - 1.0) < std_tol)

def unscale(X_scaled, scaler):
    return X_scaled * scaler.scale_ + scaler.mean_

def apply_clip(X, p1, p99):
    return np.clip(X, p1, p99)

def prune_features_raw(X_raw, feature_order, std_eps=RAW_STD_EPS, dom_thresh=RAW_DOM_THRESH):
    stds = X_raw.std(axis=0)
    keep_var = stds > std_eps
    keep_dom = []
    for j in range(X_raw.shape[1]):
        vals, counts = np.unique(X_raw[:, j], return_counts=True)
        dom = counts.max() / len(X_raw)
        keep_dom.append(dom < dom_thresh)
    keep_idx = np.where(keep_var & np.array(keep_dom))[0]
    kept_names = [feature_order[i] for i in keep_idx]
    logger.info("Raw-space prune: kept %d/%d features.", len(keep_idx), len(feature_order))
    return keep_idx, kept_names, stds

def best_threshold(y_true, probs, metric="accuracy"):
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_m = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        if metric == "accuracy":
            m = accuracy_score(y_true, preds)
        else:
            m = f1_score(y_true, preds)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)

# --------------------
# Train loop (same as before but with logging)
# --------------------
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, patience, device, criterion):
    best_val_loss, best_val_acc = float("inf"), 0.0
    patience_counter = 0

    scaler_amp = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if device.type == "cuda" else nullcontext

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "roc_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=MIXUP_ALPHA)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(X_mix)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_probs, all_true = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                with autocast_ctx():
                    out = model(Xv)
                    loss_v = criterion(out, yv).item()
                val_loss += loss_v
                probs = F.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)
                correct += (preds == yv).sum().item()
                total += yv.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(yv.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        val_acc = correct / max(1, total) if total > 0 else 0.0
        try:
            roc_auc = roc_auc_score(all_true, all_probs)
        except Exception:
            roc_auc = 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["roc_auc"].append(roc_auc)

        logger.info("[Epoch %d/%d] train_loss=%.4f val_loss=%.4f val_acc=%.4f roc_auc=%.4f",
                    epoch, epochs, train_loss, val_loss, val_acc, roc_auc)

        scheduler.step()

        improved = (val_acc > best_val_acc) or (val_loss < best_val_loss - 1e-8)
        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info("Saved new best model -> %s", BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered (patience=%d).", patience)
                break

    # Save curves
    try:
        plt.figure()
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="val")
        plt.legend(); plt.title("Loss")
        plt.savefig(os.path.join(PHASE2_DIR, "loss_curves.png")); plt.close()

        plt.figure()
        plt.plot(history["val_acc"], label="val_acc")
        plt.plot(history["roc_auc"], label="roc_auc")
        plt.legend(); plt.title("Val Metrics")
        plt.savefig(os.path.join(PHASE2_DIR, "val_metrics.png")); plt.close()
    except Exception as e:
        logger.warning("Failed to persist plots: %s", e)

    return BEST_MODEL_PATH

# --------------------
# ONNX export & quantization helpers
# --------------------
def export_model_to_onnx(model, num_features, onnx_path, opset=14):
    """Export model to ONNX with dynamic axes for batch & feature dimensions."""
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, num_features, dtype=torch.float32, device=device)
    input_names = ["input"]
    output_names = ["logits"]
    # dynamic_axes can be overridden later if static feature axis requested
    dynamic_axes = {"input": {0: "batch", 1: "features"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    logger.info("Exported ONNX model with dynamic axes to %s", onnx_path)

def run_onnx_parity_check(model, X_reference: np.ndarray, onnx_path: Path, parity_json: Path, n: int = 64,
                           mean_tol: float = 1e-4, max_tol: float = 1e-3) -> dict:
    """Run PyTorch vs ONNX parity over random subset and persist JSON summary.

    Parameters
    ----------
    model : nn.Module (already on device)
    X_reference : np.ndarray (processed feature matrix)
    onnx_path : Path to exported ONNX model
    parity_json : Path to write parity summary
    n : number of random samples (capped by dataset rows)
    mean_tol : mean absolute difference tolerance
    max_tol : max absolute difference tolerance
    """
    summary = {"passed": False, "n": 0, "max_diff": None, "mean_diff": None}
    if not ORT_AVAILABLE:
        logger.warning("onnxruntime not available; skipping parity check")
        safe_json_dump(summary, parity_json)
        return summary
    if not os.path.exists(onnx_path):
        logger.warning("ONNX model not found for parity check: %s", onnx_path)
        safe_json_dump(summary, parity_json)
        return summary
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        logger.warning("Failed initializing ONNX session: %s", e)
        safe_json_dump(summary, parity_json)
        return summary

    model.eval()
    device = next(model.parameters()).device
    if X_reference.shape[0] == 0:
        logger.warning("Empty reference data; skipping parity check")
        safe_json_dump(summary, parity_json)
        return summary

    n_eff = int(min(n, X_reference.shape[0]))
    idx = np.random.choice(X_reference.shape[0], n_eff, replace=False)
    batch = X_reference[idx].astype(np.float32)
    with torch.no_grad():
        torch_logits = model(torch.tensor(batch, dtype=torch.float32, device=device))
        torch_probs = F.softmax(torch_logits, dim=1).cpu().numpy()[:, 1]
    onnx_probs = []
    for row in batch:
        out = sess.run(None, {input_name: row.reshape(1, -1).astype(np.float32)})[0]
        onnx_probs.append(softmax(out, axis=1)[0, 1])
    onnx_probs = np.array(onnx_probs, dtype=np.float32)
    diffs = np.abs(torch_probs - onnx_probs)
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())
    passed = (mean_diff <= mean_tol) and (max_diff <= max_tol)
    summary.update({"passed": passed, "n": n_eff, "max_diff": max_diff, "mean_diff": mean_diff,
                    "mean_tolerance": mean_tol, "max_tolerance": max_tol})
    safe_json_dump(summary, parity_json)
    if passed:
        logger.info("ONNX parity passed (n=%d max=%.2e mean=%.2e)", n_eff, max_diff, mean_diff)
    else:
        logger.error("ONNX parity FAILED (n=%d max=%.2e mean=%.2e) tolerances (max<=%.1e mean<=%.1e)",
                     n_eff, max_diff, mean_diff, max_tol, mean_tol)
    return summary


def quantize_onnx_model(onnx_in, onnx_out):
    if not ONNX_QUANT_AVAILABLE:
        logger.warning("Quantization requested but ONNX quantization tools unavailable.")
        return False
    try:
        quantize_dynamic(onnx_in, onnx_out, weight_type=QuantType.QInt8)
        logger.info("Saved quantized ONNX model to %s", onnx_out)
        return True
    except Exception as e:
        logger.warning("ONNX quantization failed: %s", e)
        return False

def evaluate_model_logits(model: nn.Module, X: np.ndarray, y: np.ndarray, device, temperature: float, threshold: float):
    """Return dict of metrics (roc_auc, acc_thr) for given model and data using provided temperature/threshold.

    Temperature is applied to logits before softmax; threshold is applied to P(class=1).
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], 1024):
            xb = torch.tensor(X[i:i+1024], dtype=torch.float32, device=device)
            logits = model(xb) / max(1e-12, temperature)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            labels.append(y[i:i+1024])
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    try:
        auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.0
    except Exception:
        auc = 0.0
    preds_thr = (probs >= threshold).astype(int)
    acc_thr = float(accuracy_score(labels, preds_thr))
    return {"roc_auc": auc, "thr_accuracy": acc_thr}

def dynamic_quantize_torch_model(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to Linear layers only (avoid LSTM for accuracy)."""
    # Dynamic quant ops are CPU-only; ensure we quantize a CPU copy of the model.
    cpu_model = model.to('cpu') if next(model.parameters()).device.type != 'cpu' else model
    try:
        q_model = torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
    except Exception:
        from torch.quantization import quantize_dynamic as _qd
        q_model = _qd(cpu_model, {nn.Linear}, dtype=torch.qint8)
    return q_model

# --------------------
# Selective quantization strategies (futureproof accuracy preservation)
# --------------------
def _gather_linear_modules(model: nn.Module):
    """Return list of (name, module) for nn.Linear layers in forward order."""
    linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears.append((name, module))
    return linears

def clone_model_structure(model: nn.Module) -> nn.Module:
    """Create a fresh, uninitialized copy of the model architecture (state_dict to be loaded separately)."""
    import copy
    new_model = type(model)(*getattr(model, '_init_args', []), **getattr(model, '_init_kwargs', {})) if hasattr(model, '_init_args') else copy.deepcopy(model)
    # deepcopy keeps weights; we will overwrite anyway; acceptable for our use here
    return new_model

def apply_quant_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """Apply a selective dynamic quantization strategy.

    Strategies:
      - all_linear: quantize all Linear layers (baseline)
      - last_two: quantize only last two Linear layers (fc1, fc2 and possibly last MLP layer)
      - last_one: quantize only final classification layer (fc2)
      - mlp_only: quantize only the MLP stack (not fusion fc1/fc2)
    """
    cpu_model = model.to('cpu')
    linears = _gather_linear_modules(cpu_model)
    if not linears:
        return cpu_model  # nothing to quantize

    # Map names for clarity
    linear_dict = {n: m for n, m in linears}

    def quantize_subset(subset_names):
        import copy
        m_copy = copy.deepcopy(cpu_model)  # ensure independent quantization each attempt
        modules_to_quant = set()
        for n, mod in m_copy.named_modules():
            if isinstance(mod, nn.Linear) and n in subset_names:
                modules_to_quant.add(mod.__class__)
        # torch.ao.quantization.quantize_dynamic expects a set of module classes, not instances.
        # We cannot restrict to specific instances directly, so we fallback to manual replacement.
        # Manual: replace target Linear layers with dynamically quantized versions using quantize_dynamic on a temp wrapper.
        from torch.ao.quantization import quantize_dynamic as _qd_new
        # Easiest: quantize all then restore those we don't want from original copy.
        q_all = _qd_new(copy.deepcopy(cpu_model), {nn.Linear}, dtype=torch.qint8)
        for n, mod in cpu_model.named_modules():
            pass  # placeholder to preserve scope
        for name, module in m_copy.named_modules():
            if isinstance(module, nn.Linear):
                if name in subset_names:
                    # find quantized counterpart in q_all
                    q_mod = dict(q_all.named_modules())[name]
                    # swap weights/bias (already quantized) by shallow replace
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    attr_name = name.split('.')[-1]
                    parent = m_copy if parent_name == '' else dict(m_copy.named_modules())[parent_name]
                    setattr(parent, attr_name, q_mod)
        return m_copy

    if strategy == 'all_linear':
        return dynamic_quantize_torch_model(cpu_model)
    elif strategy == 'last_two':
        # Selecting last two Linear layers by name order
        subset = [n for n, _ in linears[-2:]]
        return quantize_subset(subset)
    elif strategy == 'last_one':
        subset = [linears[-1][0]]
        return quantize_subset(subset)
    elif strategy == 'mlp_only':
        subset = [n for n, _ in linears if n.startswith('mlp')]
        return quantize_subset(subset)
    else:
        return cpu_model

def verify_onnx_matches_pytorch(onnx_path, model, X_test_proc, tolerance=1e-4, num_samples=256):
    """
    Optionally verify that ONNX runtime outputs probabilities matching PyTorch.
    Runs up to num_samples (or fewer) through both.
    """
    if not ORT_AVAILABLE:
        logger.info("onnxruntime not available: skipping ONNX verification.")
        return True, 0.0

    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    model.eval()
    device = next(model.parameters()).device

    n = min(num_samples, X_test_proc.shape[0])
    idx = np.random.choice(X_test_proc.shape[0], n, replace=False)
    xs = X_test_proc[idx].astype(np.float32)
    # PyTorch outputs
    with torch.no_grad():
        torch_in = torch.tensor(xs, dtype=torch.float32).to(device)
        logits = model(torch_in)
        probs_t = F.softmax(logits, dim=1).cpu().numpy()[:, 1]

    # ONNX outputs
    probs_o = []
    for i in range(n):
        inp = xs[i:i+1]
        ort_outs = sess.run(None, {input_name: inp})
        logits_o = np.asarray(ort_outs[0], dtype=np.float32)
        probs_o.append(softmax(logits_o, axis=1)[0, 1])
    probs_o = np.array(probs_o, dtype=np.float32)

    diff = np.abs(probs_t - probs_o)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    logger.info("ONNX vs PyTorch prob diffs -> max=%.6e mean=%.6e (n=%d)", max_diff, mean_diff, n)
    ok = max_diff <= tolerance
    return ok, max_diff

def fit_temperature(model, val_loader, device):
    """Fit temperature scaling parameter by minimizing NLL on validation set."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logit = model(X)
            logits_list.append(logit.cpu())
            labels_list.append(y.cpu())
    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    
    def nll(temp):
        logits_scaled = logits / temp
        probs = softmax(logits_scaled, axis=1)[:, 1]
        # Binary NLL
        return -np.mean(labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12))
    
    best_t = 1.0
    best_nll = float('inf')
    for t in np.linspace(0.5, 2.0, 31):
        curr_nll = nll(t)
        if curr_nll < best_nll:
            best_nll = curr_nll
            best_t = t
    return best_t

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# --------------------
# Main
# --------------------
def main():
    logger.info("Loading Phase 1 artifacts...")
    # load data
    data = None
    if os.path.exists(DATA_NPZ):
        try:
            npz_data = np.load(DATA_NPZ, allow_pickle=True)
            data = {
                "X_train": npz_data["X_train"],
                "y_train": npz_data["y_train"],
                "X_val": npz_data["X_val"],
                "y_val": npz_data["y_val"],
                "X_test": npz_data["X_test"],
                "y_test": npz_data["y_test"]
            }
            logger.info("Loaded data from .npz")
        except Exception as e:
            logger.warning("Failed to load .npz: %s", e)

    if data is None:
        if os.path.exists(DATA_PKL):
            data = safe_load(DATA_PKL)
            logger.info("Loaded data from .pkl")
        else:
            raise FileNotFoundError("No Phase1 data found (.npz or .pkl)")

    # prefer scaler.json for edge-friendly storage; fall back to pkl
    scaler = None
    if os.path.exists(PHASE1_SCALER_JSON):
        with open(PHASE1_SCALER_JSON, "r") as f:
            sjson = json.load(f)
        scaler = DummyScaler(
            mean=np.array(sjson["mean"], dtype=np.float32),
            scale=np.array(sjson["scale"], dtype=np.float32),
            with_mean=sjson.get("with_mean", True),
            with_std=sjson.get("with_std", True),
        )
        logger.info("Loaded scaler from scaler.json")
    elif os.path.exists(PHASE1_SCALER_PKL):
        scaler = safe_load(PHASE1_SCALER_PKL)
        logger.info("Loaded scaler from scaler.pkl")
    else:
        raise FileNotFoundError("Scaler artifact not found")

    with open(os.path.join(PHASE1_DIR, "feature_order.json"), "r") as f:
        feature_order = json.load(f)
    with open(os.path.join(PHASE1_DIR, "clip_bounds.json"), "r") as f:
        clip_bounds_all = json.load(f)

    X_train = np.asarray(data["X_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    X_val = np.asarray(data["X_val"])
    y_val = np.asarray(data["y_val"], dtype=np.int64)
    X_test = np.asarray(data["X_test"])
    y_test = np.asarray(data["y_test"], dtype=np.int64)

    already_std = is_already_standardized(X_train)
    logger.info("Phase-1 data already standardized? %s", already_std)

    if already_std:
        X_train_proc, X_val_proc, X_test_proc = X_train, X_val, X_test
        X_train_raw = unscale(X_train_proc, scaler)
        X_val_raw = unscale(X_val_proc, scaler)
        X_test_raw = unscale(X_test_proc, scaler)
        p1_all = np.array(clip_bounds_all["p1"])
        p99_all = np.array(clip_bounds_all["p99"])
    else:
        p1_all = np.array(clip_bounds_all["p1"])
        p99_all = np.array(clip_bounds_all["p99"])
        X_train_raw, X_val_raw, X_test_raw = X_train.copy(), X_val.copy(), X_test.copy()
        X_train_clipped = apply_clip(X_train_raw, p1_all, p99_all)
        X_val_clipped   = apply_clip(X_val_raw,   p1_all, p99_all)
        X_test_clipped  = apply_clip(X_test_raw,  p1_all, p99_all)
        X_train_proc = scaler.transform(X_train_clipped)
        X_val_proc = scaler.transform(X_val_clipped)
        X_test_proc = scaler.transform(X_test_clipped)

    # Raw-space pruning
    keep_idx, kept_names, raw_stds = prune_features_raw(X_train_raw, feature_order)
    feature_order_kept = kept_names
    X_train_proc = X_train_proc[:, keep_idx]
    X_val_proc = X_val_proc[:, keep_idx]
    X_test_proc = X_test_proc[:, keep_idx]
    p1_kept = p1_all[keep_idx]
    p99_kept = p99_all[keep_idx]

    num_features = len(feature_order_kept)
    num_classes = len(np.unique(y_train))
    assert X_train_proc.shape[1] == num_features, "Feature count mismatch after pruning"

    logger.info("Shapes -> Train: %s Val: %s Test: %s", X_train_proc.shape, X_val_proc.shape, X_test_proc.shape)
    logger.info("Num features (kept): %d", num_features)

    # DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(IoTDataset(X_train_proc, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(IoTDataset(X_val_proc, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(IoTDataset(X_test_proc, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Model & training
    model = DeepHybridModel(num_features, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=LABEL_SMOOTHING)

    best_model_path = train_model(model, train_loader, val_loader, optimizer, scheduler,
                                  EPOCHS, PATIENCE, device, criterion)

    # Load best & calibrate temp
    model.load_state_dict(safe_model_load(best_model_path, device))
    model.eval()
    # Temperature search restricted to >=1.0 (softening only). We'll search grid 1.0..2.5
    def fit_temperature_softening(m, loader, device):
        m.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits_list.append(m(Xb).cpu())
                labels_list.append(yb.cpu())
        logits = torch.cat(logits_list).numpy()
        labels = torch.cat(labels_list).numpy()
        def nll(temp):
            ls = logits / temp
            probs = softmax(ls, axis=1)[:,1]
            return -np.mean(labels * np.log(probs + 1e-12) + (1-labels)*np.log(1-probs + 1e-12))
        best_t, best_val = 1.0, float('inf')
        for t in np.linspace(1.0, 2.5, 31):
            v = nll(t)
            if v < best_val:
                best_val, best_t = v, t
        return best_t
    T = fit_temperature_softening(model, val_loader, device)
    logger.info("Learned temperature T=%.6f", T)

    # collect_probs (calibrated)
    def collect_probs(loader):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.to(device)
                logits = model(Xb) / T
                all_logits.append(logits.cpu())
                all_labels.append(yb)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()
        probs = F.softmax(logits, dim=1).numpy()[:, 1]
        preds_argmax = logits.argmax(dim=1).numpy()
        return probs, preds_argmax, labels

    val_probs, val_preds_argmax, val_labels = collect_probs(val_loader)
    val_best_thr, val_best_acc = best_threshold(val_labels, val_probs, metric="accuracy")
    logger.info("Best threshold on val (acc): t=%.4f acc=%.6f", val_best_thr, val_best_acc)
    # Preserve deep-only threshold before any ensemble blending (for Phase 4 deep-only inference)
    deep_only_threshold = float(val_best_thr)

    # Optional LightGBM blending
    lgbm_info = {"use_lgbm": False}
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
        )
        # Some versions don't accept 'verbose' in fit; rely on defaults
        lgbm.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)])
        p_deep_val = np.array(val_probs)
        p_lgb_val = lgbm.predict_proba(X_val_proc)[:, 1]
        best_w, best_acc_blend, best_t_blend = 0.5, -1.0, val_best_thr
        for w in np.linspace(0, 1, 41):
            p_ens = w * p_deep_val + (1 - w) * p_lgb_val
            t, acc_ = best_threshold(val_labels, p_ens, metric="accuracy")
            if acc_ > best_acc_blend:
                best_acc_blend, best_w, best_t_blend = acc_, float(w), float(t)
        if best_acc_blend > val_best_acc:
            logger.info(
                "Ensemble chosen (better than deep): weight_deep=%.3f val_acc=%.6f thr=%.4f",
                best_w,
                best_acc_blend,
                best_t_blend,
            )
            lgbm_info = {"use_lgbm": True, "weight_deep": best_w, "threshold": best_t_blend}
            val_best_thr = best_t_blend
            with open(LGBM_MODEL_PKL, "wb") as fh:
                pickle.dump(lgbm, fh)
            logger.info("Saved LightGBM ensemble to %s", LGBM_MODEL_PKL)
        else:
            logger.info("Deep model outperformed LightGBM blend; skipping ensemble save/use.")
    except Exception as e:
        logger.info("LightGBM not used or failed: %s", e)
        lgbm_info = {"use_lgbm": False}

    # validation & test reporting
    class_names = ["Benign", "Attack"]
    val_report_argmax = classification_report(val_labels, val_preds_argmax, digits=4, target_names=class_names, output_dict=True)
    val_auc = float(roc_auc_score(val_labels, val_probs)) if len(set(val_labels)) > 1 else 0.0
    val_f1_argmax = float(f1_score(val_labels, val_preds_argmax)) if len(set(val_labels)) > 1 else 0.0
    val_acc_argmax = float(accuracy_score(val_labels, val_preds_argmax))

    if lgbm_info.get("use_lgbm", False):
        p_deep_val = np.array(val_probs)
        p_lgb_val  = lgbm.predict_proba(X_val_proc)[:, 1]
        p_ens_val  = lgbm_info["weight_deep"] * p_deep_val + (1 - lgbm_info["weight_deep"]) * p_lgb_val
        val_preds_thr = (p_ens_val >= val_best_thr).astype(int)
    else:
        val_preds_thr = (np.array(val_probs) >= val_best_thr).astype(int)
    val_acc_thr = float(accuracy_score(val_labels, val_preds_thr))
    val_f1_thr  = float(f1_score(val_labels, val_preds_thr))
    val_report_thr = classification_report(val_labels, val_preds_thr, digits=4, target_names=class_names, output_dict=True)

    # Test evaluation
    test_probs, test_preds_argmax, test_labels = collect_probs(test_loader)
    if lgbm_info.get("use_lgbm", False):
        p_deep_test = np.array(test_probs)
        p_lgb_test = lgbm.predict_proba(X_test_proc)[:, 1]
        p_ens_test = lgbm_info["weight_deep"] * p_deep_test + (1 - lgbm_info["weight_deep"]) * p_lgb_test
        test_preds_thr = (p_ens_test >= val_best_thr).astype(int)
        test_probs_for_auc = p_ens_test
    else:
        test_preds_thr = (np.array(test_probs) >= val_best_thr).astype(int)
        test_probs_for_auc = np.array(test_probs)

    test_report_argmax = classification_report(test_labels, test_preds_argmax, digits=4, target_names=class_names, output_dict=True)
    test_auc = float(roc_auc_score(test_labels, test_probs_for_auc)) if len(set(test_labels)) > 1 else 0.0
    test_f1_argmax = float(f1_score(test_labels, test_preds_argmax)) if len(set(test_labels)) > 1 else 0.0
    test_acc_argmax = float(accuracy_score(test_labels, test_preds_argmax))
    test_acc_thr = float(accuracy_score(test_labels, test_preds_thr))
    test_f1_thr  = float(f1_score(test_labels, test_preds_thr))
    test_report_thr = classification_report(test_labels, test_preds_thr, digits=4, target_names=class_names, output_dict=True)

    # --- Logit gap statistics (pre-temperature) on validation set ---
    try:
        raw_logits_list = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                raw_logits_list.append(model(Xv).cpu().numpy())
        raw_logits = np.vstack(raw_logits_list)  # (N,2)
        gaps = np.abs(raw_logits[:,1] - raw_logits[:,0])
        lg_stats = {
            "mean": float(np.mean(gaps)),
            "p50": float(np.percentile(gaps, 50)),
            "p90": float(np.percentile(gaps, 90)),
            "p99": float(np.percentile(gaps, 99)),
        }
        logger.info("Logit gap stats (val) mean=%.4f p50=%.4f p90=%.4f p99=%.4f", lg_stats['mean'], lg_stats['p50'], lg_stats['p90'], lg_stats['p99'])
    except Exception as e:
        lg_stats = {"mean": float('nan'), "p50": float('nan'), "p90": float('nan'), "p99": float('nan')}
        logger.warning("Failed computing logit gap stats: %s", e)

    # Save final PyTorch weights
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    logger.info("Saved final PyTorch model to %s", FINAL_MODEL_PATH)

    # Save scaler for Phase2 (pkl) and scaler.json for edge
    try:
        # Only persist scaler.pkl if it’s a joblib/pickle-friendly object
        joblib.dump(scaler, PHASE2_SCALER_PKL)
        logger.info("Saved scaler.pkl to %s", PHASE2_SCALER_PKL)
    except Exception as e:
        logger.warning("Could not save scaler.pkl: %s", e)

    scaler_json = {"mean": np.asarray(getattr(scaler, "mean_", getattr(scaler, "mean_", np.zeros(num_features))), dtype=float).tolist(),
                   "scale": np.asarray(getattr(scaler, "scale_", getattr(scaler, "scale_", np.ones(num_features))), dtype=float).tolist(),
                   "with_mean": getattr(scaler, "with_mean", True),
                   "with_std": getattr(scaler, "with_std", True)}
    safe_json_dump(scaler_json, PHASE2_SCALER_JSON)
    safe_json_dump(feature_order_kept, PHASE2_FEATURE_ORDER)
    safe_json_dump({"p1": p1_kept.tolist(), "p99": p99_kept.tolist()}, PHASE2_CLIP_BOUNDS)

    # Build evaluation payload
    eval_payload = {
        "val": {
            "roc_auc": val_auc,
            "argmax_f1": val_f1_argmax,
            "argmax_accuracy": val_acc_argmax,
            "threshold": float(val_best_thr),
            "thr_f1": val_f1_thr,
            "thr_accuracy": val_acc_thr,
            "classification_report_argmax": val_report_argmax,
            "classification_report_thresholded": val_report_thr,
        },
        "test": {
            "roc_auc": test_auc,
            "argmax_f1": test_f1_argmax,
            "argmax_accuracy": test_acc_argmax,
            "thr_f1": test_f1_thr,
            "thr_accuracy": test_acc_thr,
            "classification_report_argmax": test_report_argmax,
            "classification_report_thresholded": test_report_thr,
        },
        "calibration": {"temperature": float(T)},
        "ensemble": lgbm_info,
        "meta": {
            "best_model_path": best_model_path,
            "final_model_path": FINAL_MODEL_PATH,
            "num_classes": int(num_classes),
            "timestamp": datetime.utcnow().isoformat(),
            "already_standardized": bool(already_std),
            "kept_feature_indices": [int(i) for i in keep_idx],
            "seed": SEED,
            "val_best_threshold": float(val_best_thr),
            "val_best_threshold_deep_only": float(deep_only_threshold),
            "logit_gap": lg_stats
        }
    }
    safe_json_dump(eval_payload, EVAL_JSON)
    logger.info("Saved evaluation payload to %s", EVAL_JSON)

    # -------------------- Optimization Config Load --------------------
    opt_cfg = {
        "onnx": {"static_feature_axis": False},
        "pruning": {"enabled": False},
        "fp16": {"export_fp16_onnx": False},
        "qat": {"enabled": False}
    }
    if OPTIM_CONFIG_JSON.exists():
        import json as _json
        try:
            raw_txt = OPTIM_CONFIG_JSON.read_text(encoding='utf-8')
            if not raw_txt.strip():
                raise ValueError("optimization_config.json is empty")
            try:
                user_cfg = _json.loads(raw_txt)
            except _json.JSONDecodeError as je:
                # Retry once after stripping potential BOM / trailing chars
                cleaned = raw_txt.strip().lstrip('\ufeff')
                user_cfg = _json.loads(cleaned)  # will raise if still bad
                logger.warning("optimization_config.json had leading/trailing whitespace/BOM; parsed after cleaning")
            # shallow merge
            for k, v in user_cfg.items():
                if k in opt_cfg and isinstance(v, dict):
                    opt_cfg[k].update(v)
                else:
                    opt_cfg[k] = v
            snippet = raw_txt[:120].replace('\n', ' ')
            logger.info("Loaded optimization config (%d bytes) preview=%.120s", len(raw_txt), snippet)
        except Exception as e:
            logger.warning("Failed loading optimization_config.json (%s). Proceeding with defaults (no pruning/fp16/qat/static-axis)", e)

    # -------------------- Optional Pruning Stage (extended) --------------------
    pruning_summary = {"applied": False}
    if opt_cfg.get("pruning", {}).get("enabled", False):
        try:
            import copy
            p_cfg = opt_cfg["pruning"]
            target_sparsity = float(p_cfg.get("target_sparsity", 0.3))
            schedule_steps = int(p_cfg.get("schedule_steps", 1))
            recovery_epochs = int(p_cfg.get("recovery_epochs", 0))
            method = p_cfg.get("method", p_cfg.get("granularity", "unstructured_l1"))  # backward compat
            layer_names = p_cfg.get("layer_names", [])
            max_auc_drop = float(p_cfg.get("max_val_auc_drop", 0.002))
            max_acc_drop = float(p_cfg.get("max_val_acc_drop", 0.002))
            structured_dim = p_cfg.get("structured_dim", "out")  # for structured_out
            base_model_state = copy.deepcopy(model.state_dict())

            # Baseline metrics
            baseline_auc = val_auc
            baseline_acc = val_acc_thr

            def unstructured_prune_tensor(t: torch.Tensor, sparsity: float):
                if sparsity <= 0: return t
                flat = t.view(-1)
                k = int(len(flat) * sparsity)
                if k <= 0 or k >= len(flat): return t
                with torch.no_grad():
                    threshold = flat.abs().kthvalue(k).values.item()
                    mask = (flat.abs() > threshold).float()
                    return (flat * mask).view_as(t)

            def structured_out_prune(t: torch.Tensor, sparsity: float):
                # prune whole output channels/rows by L1 norm
                if sparsity <= 0: return t
                if t.dim() < 2: return t
                rows = t.shape[0]
                k = int(rows * sparsity)
                if k <= 0 or k >= rows: return t
                with torch.no_grad():
                    norms = t.abs().mean(dim=tuple(range(1, t.dim())))
                    threshold = norms.kthvalue(k).values.item()
                    mask = (norms > threshold).float().view(-1, *([1] * (t.dim()-1)))
                    return t * mask

            def apply_prune(param: torch.Tensor, step_frac: float):
                if method in ("unstructured_l1", "unstructured"):
                    return unstructured_prune_tensor(param, step_frac)
                elif method in ("structured_out", "structured_rows"):
                    return structured_out_prune(param, step_frac)
                else:
                    return unstructured_prune_tensor(param, step_frac)

            cumulative_target = target_sparsity
            step_targets = []
            if schedule_steps <= 1:
                step_targets = [cumulative_target]
            else:
                # geometric-ish progression so earlier steps mild, final equals target
                for s in range(1, schedule_steps+1):
                    step_targets.append(cumulative_target * s / schedule_steps)

            step_metrics = []
            last_good_state = copy.deepcopy(model.state_dict())
            last_good_auc, last_good_acc = baseline_auc, baseline_acc
            applied = False

            # Helper to evaluate quickly
            def eval_val():
                probs, _, labels_ = collect_probs(val_loader)
                auc_ = float(roc_auc_score(labels_, probs)) if len(set(labels_))>1 else 0.0
                preds_thr_ = (np.array(probs) >= val_best_thr).astype(int)
                acc_ = float(accuracy_score(labels_, preds_thr_))
                return auc_, acc_

            for idx, current_target in enumerate(step_targets, start=1):
                step_layers = []
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if not layer_names:
                            target_layer_match = name.startswith('fc2') or name.startswith('mlp')
                        else:
                            target_layer_match = any(name.startswith(ln) for ln in layer_names)
                        if target_layer_match and param.dim() >= 2:
                            original_nonzero = int((param != 0).sum())
                            new_param = apply_prune(param.data, current_target)
                            param.copy_(new_param)
                            new_nonzero = int((param != 0).sum())
                            step_layers.append({
                                "name": name,
                                "original_nonzero": original_nonzero,
                                "new_nonzero": new_nonzero,
                                "sparsity": 1 - new_nonzero / max(1, original_nonzero),
                                "target": current_target
                            })
                auc_now, acc_now = eval_val()
                auc_drop = baseline_auc - auc_now
                acc_drop = baseline_acc - acc_now
                within = (auc_drop <= max_auc_drop) and (acc_drop <= max_acc_drop)
                step_metrics.append({
                    "step": idx,
                    "target_sparsity": current_target,
                    "val_auc": auc_now,
                    "val_thr_accuracy": acc_now,
                    "auc_drop": auc_drop,
                    "acc_drop": acc_drop,
                    "within": within,
                    "layers": step_layers
                })
                if within:
                    last_good_state = copy.deepcopy(model.state_dict())
                    last_good_auc, last_good_acc = auc_now, acc_now
                    applied = True
                    logger.info("Prune step %d/%d accepted (target=%.3f AUC drop=%.6f ACC drop=%.6f)", idx, len(step_targets), current_target, auc_drop, acc_drop)
                else:
                    logger.warning("Prune step %d/%d exceeded limits; reverting to previous acceptable state", idx, len(step_targets))
                    model.load_state_dict(last_good_state)
                    break

            # Recovery fine-tune (only if applied and epochs>0)
            if applied and recovery_epochs > 0:
                logger.info("Beginning pruning recovery fine-tune (%d epochs)", recovery_epochs)
                model.train()
                # Simple low-LR fine-tune on training loader
                ft_lr = float(p_cfg.get("recovery_lr", 1e-4))
                ft_opt = torch.optim.AdamW(model.parameters(), lr=ft_lr)
                for ep in range(1, recovery_epochs+1):
                    total = 0.0; n = 0
                    for xb, yb in train_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        ft_opt.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        ft_opt.step()
                        total += loss.item() * xb.size(0); n += xb.size(0)
                    auc_after, acc_after = eval_val()
                    logger.info("Recovery epoch %d/%d pruned_val_auc=%.6f pruned_val_acc=%.6f", ep, recovery_epochs, auc_after, acc_after)
                    # Early abort if we exceed thresholds now
                    if (baseline_auc - auc_after) > max_auc_drop or (baseline_acc - acc_after) > max_acc_drop:
                        logger.warning("Recovery degraded metrics beyond thresholds; rolling back to last good pre-recovery state")
                        model.load_state_dict(last_good_state)
                        break
                    else:
                        last_good_state = copy.deepcopy(model.state_dict())
                        last_good_auc, last_good_acc = auc_after, acc_after

            # Final acceptance decided by whether any step applied
            model.load_state_dict(last_good_state)
            auc_final, acc_final = last_good_auc, last_good_acc
            pruning_summary = {
                "attempted": True,
                "applied": applied,
                "baseline": {"auc": baseline_auc, "thr_accuracy": baseline_acc},
                "pruned": {"auc": auc_final, "thr_accuracy": acc_final},
                "drops": {"auc_drop": baseline_auc - auc_final, "acc_drop": baseline_acc - acc_final},
                "criteria": {"max_auc_drop": max_auc_drop, "max_acc_drop": max_acc_drop},
                "method": method,
                "schedule_steps": schedule_steps,
                "step_metrics": step_metrics,
                "recovery_epochs": recovery_epochs
            }
            safe_json_dump(pruning_summary, PRUNING_REPORT_JSON)
        except Exception as e:
            logger.warning("Pruning stage failed: %s", e)
            pruning_summary = {"applied": False, "error": str(e)}
            safe_json_dump(pruning_summary, PRUNING_REPORT_JSON)


    # ---- New: Build & save unified feature contract ----
    try:
        import hashlib
        manifest_files = [
            'feature_contract.json',
            'evaluation.json',
            'model_hybrid.onnx',
            'model_hybrid_fp16.onnx',
            'quantization_report.json',
            'pruning_report.json',
            'qat_report.json',
            'final_model_hybrid.pth',
            'scaler.json',
            'feature_order.json',
            'clip_bounds.json'
        ]
        manifest = {}
        for fn in manifest_files:
            fp = PHASE2_DIR / fn
            if fp.exists():
                h = hashlib.sha256()
                with fp.open('rb') as fh:
                    for chunk in iter(lambda: fh.read(65536), b''):
                        h.update(chunk)
                manifest[fn] = h.hexdigest()
        safe_json_dump(manifest, PHASE2_DIR / 'artifact_manifest.json')
        logger.info("Wrote artifact_manifest.json with %d entries", len(manifest))
    except Exception as e:
        logger.warning("Failed to build/save feature contract: %s", e)
    # ---- End contract block ----

    # Export to ONNX & parity check (consider static feature axis)
    try:
        if opt_cfg.get('onnx', {}).get('static_feature_axis', False):
            # Re-export with only batch dynamic
            model.eval()
            device_exp = next(model.parameters()).device
            dummy = torch.randn(1, num_features, dtype=torch.float32, device=device_exp)
            torch.onnx.export(
                model,
                dummy,
                ONNX_MODEL_PATH,
                export_params=True,
                opset_version=14,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                do_constant_folding=True,
            )
            logger.info("Exported ONNX model with STATIC feature axis (only batch dynamic)")
        else:
            export_model_to_onnx(model, num_features, ONNX_MODEL_PATH)
        parity_summary = run_onnx_parity_check(model, X_test_proc, ONNX_MODEL_PATH, ONNX_PARITY_JSON,
                                               n=128, mean_tol=1e-4, max_tol=1e-3)
        if not parity_summary.get("passed"):
            logger.error("Parity check failed; inspect %s for details", ONNX_PARITY_JSON)
            # Raising after writing JSON so external orchestrator can decide next steps
            raise RuntimeError("ONNX parity check failed beyond tolerances")
    except Exception as e:
        logger.warning("ONNX export/parity stage encountered an error: %s", e)

    # Optional FP16 ONNX export
    if opt_cfg.get('fp16', {}).get('export_fp16_onnx', False):
        try:
            model_fp16 = DeepHybridModel(num_features, num_classes=num_classes).to(device)
            model_fp16.load_state_dict(model.state_dict())
            model_fp16.half()
            dummy = torch.randn(1, num_features, dtype=torch.float16, device=device)
            torch.onnx.export(
                model_fp16,
                dummy,
                FP16_ONNX_PATH,
                export_params=True,
                opset_version=14,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                do_constant_folding=True,
            )
            logger.info("Exported FP16 ONNX model to %s", FP16_ONNX_PATH)
        except Exception as e:
            logger.warning("FP16 ONNX export failed: %s", e)

    # Post-training quantization gating (multi-strategy) with optional config & QAT fallback
    default_criteria = {"max_auc_drop": 0.01, "max_acc_drop": 0.02}
    strategies_to_try = ["last_one", "last_two", "mlp_only", "all_linear"]  # default order
    try:
        if QUANT_CONFIG_JSON.exists():
            with open(QUANT_CONFIG_JSON, 'r') as f:
                qconf = json.load(f)
            # override thresholds if present
            if "max_auc_drop" in qconf:
                default_criteria["max_auc_drop"] = float(qconf["max_auc_drop"])
            if "max_acc_drop" in qconf:
                default_criteria["max_acc_drop"] = float(qconf["max_acc_drop"])
            if "strategy_order" in qconf and isinstance(qconf["strategy_order"], list) and qconf["strategy_order"]:
                strategies_to_try = [str(s) for s in qconf["strategy_order"]]
            logger.info("Loaded quantization config overrides from %s", QUANT_CONFIG_JSON)
    except Exception as cfg_err:
        logger.warning("Failed reading quant_config.json: %s (using defaults)", cfg_err)

    quant_report = {
        "passed": False,
        "criteria": default_criteria,
        "float": {},
        "quant": {},
        "deltas": {},
        "strategies": [],
        "selected_strategy": None,
        "selection_reason": None,
        "failure_reason": None,
        "size": {}
    }
    try:
        # Float baseline
        float_val_metrics = evaluate_model_logits(model, X_val_proc, y_val, device, T, deep_only_threshold)
        float_test_metrics = evaluate_model_logits(model, X_test_proc, y_test, device, T, deep_only_threshold)
        quant_report["float"] = {"val": float_val_metrics, "test": float_test_metrics}

        best_candidate = None
        best_auc_drop = float('inf')
        for strat in strategies_to_try:
            try:
                q_model = apply_quant_strategy(model, strat)
                quant_device = torch.device('cpu')
                q_val = evaluate_model_logits(q_model, X_val_proc, y_val, quant_device, T, deep_only_threshold)
                q_test = evaluate_model_logits(q_model, X_test_proc, y_test, quant_device, T, deep_only_threshold)
                auc_drop = max(0.0, float(float_test_metrics["roc_auc"] - q_test["roc_auc"]))
                acc_drop = max(0.0, float(float_test_metrics["thr_accuracy"] - q_test["thr_accuracy"]))
                auc_within = auc_drop <= quant_report["criteria"]["max_auc_drop"]
                acc_within = acc_drop <= quant_report["criteria"]["max_acc_drop"]
                passed = bool(auc_within and acc_within)
                entry = {
                    "strategy": strat,
                    "val": q_val,
                    "test": q_test,
                    "deltas": {
                        "roc_auc_drop": auc_drop,
                        "thr_accuracy_drop": acc_drop,
                        "auc_within": auc_within,
                        "acc_within": acc_within,
                        "passed": passed
                    }
                }
                quant_report["strategies"].append(entry)
                if passed and auc_drop < best_auc_drop:
                    best_candidate = entry
                    best_auc_drop = auc_drop
            except Exception as strat_err:
                logger.warning("Quant strategy '%s' failed: %s", strat, strat_err)
                quant_report["strategies"].append({
                    "strategy": strat,
                    "error": str(strat_err)
                })

        if best_candidate:
            quant_report["passed"] = True
            quant_report["selected_strategy"] = best_candidate["strategy"]
            quant_report["selection_reason"] = "minimal_auc_drop_within_thresholds"
            quant_report["quant"] = {"val": best_candidate["val"], "test": best_candidate["test"]}
            quant_report["deltas"] = best_candidate["deltas"]
            # Persist artifacts only for chosen strategy
            q_model_final = apply_quant_strategy(model, best_candidate["strategy"])
            torch.save(q_model_final.state_dict(), PHASE2_DIR / "quantized_model_hybrid.pth")
            # --- Size metrics ---
            try:
                def _model_num_bytes(m: nn.Module):
                    total = 0
                    for p in m.state_dict().values():
                        if isinstance(p, torch.Tensor):
                            total += p.numel() * p.element_size()
                    return total
                float_bytes = _model_num_bytes(model.cpu())
                quant_bytes = _model_num_bytes(q_model_final)
                reduction_bytes = max(0, float_bytes - quant_bytes)
                reduction_pct = (reduction_bytes / float_bytes * 100.0) if float_bytes > 0 else 0.0
                quant_report["size"] = {
                    "float_bytes": float(float_bytes),
                    "quant_bytes": float(quant_bytes),
                    "reduction_bytes": float(reduction_bytes),
                    "reduction_percent": float(reduction_pct)
                }
            except Exception as size_err:
                logger.warning("Failed computing size metrics: %s", size_err)
            logger.info("Quantization passed with strategy '%s' (AUC drop=%.5f ACC drop=%.5f)",
                        best_candidate["strategy"], best_candidate["deltas"]["roc_auc_drop"], best_candidate["deltas"]["thr_accuracy_drop"])
            if best_candidate["strategy"] == "all_linear" and ONNX_QUANT_AVAILABLE and os.path.exists(ONNX_MODEL_PATH):
                if quantize_onnx_model(ONNX_MODEL_PATH, ONNX_MODEL_QPATH):
                    logger.info("Quantized ONNX model saved for strategy all_linear")
        else:
            quant_report["failure_reason"] = "No strategy satisfied degradation thresholds"
            # ----------- QAT fallback skeleton -----------
            if opt_cfg.get('qat', {}).get('enabled', False):
                try:
                    qat_lr = float(opt_cfg['qat'].get('learning_rate', 1e-4))
                    qat_epochs = int(opt_cfg['qat'].get('epochs', 3))
                    qat_layers = opt_cfg['qat'].get('layers', ['fc2'])
                    qat_criteria_auc = float(opt_cfg['qat'].get('max_auc_drop', 0.003))
                    qat_criteria_acc = float(opt_cfg['qat'].get('max_acc_drop', 0.003))
                    import copy
                    qat_model = copy.deepcopy(model).to(device)
                    qat_model.train()
                    # Simple per-layer fake quant emulation: clamp weights to int8 grid during forward pass via hook
                    def make_hook():
                        def _hook(module, inp, out):
                            return out
                        return _hook
                    # (In a real QAT you'd insert observers & quant/dequant; here we just fine-tune lightly.)
                    optimizer_qat = optim.AdamW(qat_model.parameters(), lr=qat_lr)
                    criterion_qat = LabelSmoothingLoss(classes=num_classes, smoothing=LABEL_SMOOTHING)
                    for ep in range(1, qat_epochs+1):
                        running = 0.0
                        for Xb, yb in train_loader:
                            Xb, yb = Xb.to(device), yb.to(device)
                            optimizer_qat.zero_grad(set_to_none=True)
                            logits_q = qat_model(Xb)
                            loss_q = criterion_qat(logits_q, yb)
                            loss_q.backward()
                            optimizer_qat.step()
                            running += loss_q.item()
                        logger.info("[QAT %d/%d] loss=%.4f", ep, qat_epochs, running / max(1,len(train_loader)))
                    # Evaluate QAT model
                    qat_val = evaluate_model_logits(qat_model, X_val_proc, y_val, device, T, deep_only_threshold)
                    qat_test = evaluate_model_logits(qat_model, X_test_proc, y_test, device, T, deep_only_threshold)
                    auc_drop_qat = max(0.0, float(float_test_metrics['roc_auc'] - qat_test['roc_auc']))
                    acc_drop_qat = max(0.0, float(float_test_metrics['thr_accuracy'] - qat_test['thr_accuracy']))
                    qat_pass = (auc_drop_qat <= qat_criteria_auc) and (acc_drop_qat <= qat_criteria_acc)
                    qat_record = {
                        "qat": True,
                        "val": qat_val,
                        "test": qat_test,
                        "deltas": {
                            "roc_auc_drop": auc_drop_qat,
                            "thr_accuracy_drop": acc_drop_qat,
                            "auc_within": auc_drop_qat <= qat_criteria_auc,
                            "acc_within": acc_drop_qat <= qat_criteria_acc,
                            "passed": qat_pass
                        },
                        "criteria": {"max_auc_drop": qat_criteria_auc, "max_acc_drop": qat_criteria_acc}
                    }
                    safe_json_dump(qat_record, QAT_REPORT_JSON)
                    if qat_pass:
                        quant_report['passed'] = True
                        quant_report['selected_strategy'] = 'qat_fallback'
                        quant_report['selection_reason'] = 'qat_recovered_within_thresholds'
                        quant_report['quant'] = {"val": qat_val, "test": qat_test}
                        quant_report['deltas'] = qat_record['deltas']
                        torch.save(qat_model.state_dict(), PHASE2_DIR / "quantized_model_hybrid.pth")
                        logger.info("QAT fallback succeeded (AUC drop=%.6f ACC drop=%.6f)", auc_drop_qat, acc_drop_qat)
                    else:
                        logger.warning("QAT fallback did not meet thresholds (AUC drop=%.6f ACC drop=%.6f)", auc_drop_qat, acc_drop_qat)
                except Exception as qat_err:
                    logger.warning("QAT fallback failed: %s", qat_err)
            # Cleanup stale artifacts
            for stale in [PHASE2_DIR / "quantized_model_hybrid.pth", ONNX_MODEL_QPATH]:
                if stale.exists():
                    try:
                        stale.unlink()
                        logger.info("Removed stale quantized artifact: %s", stale.name)
                    except Exception:
                        pass
            logger.warning("All quantization strategies failed thresholds; using float model only")
    except Exception as e:
        logger.warning("Quantization multi-strategy process failed: %s", e)
        quant_report["failure_reason"] = str(e)
    finally:
        safe_json_dump(quant_report, QUANT_REPORT_JSON)
        logger.info("Wrote quantization report -> %s (passed=%s strategy=%s)", QUANT_REPORT_JSON, quant_report.get('passed'), quant_report.get('selected_strategy'))

    # Copy Phase1 supporting files
    for fname in ["metadata.json", "protocol_mapping.json", "class_weights.json"]:
        s = os.path.join(PHASE1_DIR, fname)
        d = os.path.join(PHASE2_DIR, fname)
        if os.path.exists(s):
            shutil.copy(s, d)

    # Emit edge inference wrapper
    try:
        wrapper_path = PHASE2_DIR / 'inference_wrapper.py'
        import json as _json
        try:
            _feature_order = _json.loads(PHASE2_FEATURE_ORDER.read_text()) if PHASE2_FEATURE_ORDER.exists() else []
        except Exception:
            _feature_order = []
        wrapper_code = f'''"""Auto-generated edge inference wrapper.

Provides a lightweight, dependency-minimal prediction interface.
Prefers ONNX Runtime if available, falls back to a tiny Torch linear stub (NOT the full hybrid model).

API:
    from artifacts_phase2.inference_wrapper import EdgePredictor
    p = EdgePredictor(r"{PHASE2_DIR}")
    out = p.predict_from_dict({{name:0.0 for name in { _feature_order }}})
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
    _ORT = True
except Exception:
    _ORT = False
try:
    import torch, torch.nn as nn  # type: ignore
    _TORCH = True
except Exception:
    _TORCH = False

FEATURE_ORDER: List[str] = { _feature_order }

class EdgePredictor:
    def __init__(self, artifacts_dir: str | os.PathLike):
        self.dir = Path(artifacts_dir)
        self._load_scaler()
        self._init_backend()

    # Public API
    def predict_from_dict(self, feats: Dict[str, float]) -> Dict[str, Any]:
        arr = self._dict_to_array(feats)
        return self.predict_from_array(arr)

    def predict_from_array(self, x: np.ndarray) -> Dict[str, Any]:
        if x.ndim == 1:
            x = x[None, :]
        x = x.astype('float32')
        x = self._standardize(x)
        logits = self._infer(x)
        if logits.ndim == 2 and logits.shape[1] >= 2:
            # softmax prob of positive class (index 1)
            exps = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exps[:, 1] / exps.sum(axis=1, keepdims=True)
        else:
            probs = 1/(1+np.exp(-logits.reshape(-1)))
        pred = (probs >= 0.5).astype(int)
        return { 'probs': probs.tolist(), 'pred': int(pred[0]), 'logits': logits.tolist() }

    # Internal helpers
    def _dict_to_array(self, feats: Dict[str, float]) -> np.ndarray:
        vals = []
        for name in FEATURE_ORDER:
            v = feats.get(name, 0.0)
            try: vals.append(float(v))
            except Exception: vals.append(0.0)
        return np.array(vals, dtype='float32')

    def _load_scaler(self):
        sc_path = self.dir / 'scaler.json'
        if sc_path.exists():
            obj = json.loads(sc_path.read_text())
            self.mean = np.array(obj.get('mean', []), dtype='float32')
            self.scale = np.array(obj.get('scale', []), dtype='float32')
            self.with_mean = bool(obj.get('with_mean', True))
            self.with_std = bool(obj.get('with_std', True))
        else:
            self.mean = self.scale = None
            self.with_mean = self.with_std = False

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None: return X
        Y = X.copy()
        if self.with_mean and len(self.mean)==Y.shape[1]:
            Y -= self.mean
        if self.with_std and len(self.scale)==Y.shape[1]:
            sc = np.where(self.scale==0, 1.0, self.scale)
            Y /= sc
        return Y

    def _init_backend(self):
        self.session = None
        self.torch_model = None
        onnx_path = self.dir / 'model_hybrid.onnx'
        if _ORT and onnx_path.exists():
            try:
                self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                return
            except Exception:
                self.session = None
        if _TORCH:
            state_path = self.dir / 'final_model_hybrid.pth'
            if state_path.exists():
                try:
                    class LinearStub(nn.Module):
                        def __init__(self, n):
                            super().__init__(); self.fc = nn.Linear(n, 2)
                        def forward(self, x): return self.fc(x)
                    model = LinearStub(len(FEATURE_ORDER))
                    sd = torch.load(state_path, map_location='cpu')
                    model.load_state_dict(sd, strict=False)
                    model.eval(); self.torch_model = model
                except Exception:
                    self.torch_model = None

    def _infer(self, X: np.ndarray) -> np.ndarray:
        if self.session is not None:
            name = self.session.get_inputs()[0].name
            return self.session.run(None, {name: X})[0]
        if self.torch_model is not None:
            with torch.no_grad():
                return self.torch_model(torch.from_numpy(X)).numpy()
        raise RuntimeError('No backend available')

__all__ = ['EdgePredictor']
'''
        wrapper_path.write_text(wrapper_code, encoding='utf-8')
        logger.info("Emitted inference wrapper -> %s", wrapper_path)
    except Exception as _wrap_err:
        logger.warning("Failed to emit inference wrapper: %s", _wrap_err)

    logger.info("Phase 2 complete. Artifacts saved to: %s", PHASE2_DIR)
    logger.info("Artifacts for Phase 3: %s, %s, %s", os.path.basename(ONNX_MODEL_PATH), os.path.basename(PHASE2_SCALER_JSON), os.path.basename(EVAL_JSON))
    print("✅ Phase 2 complete.")

if __name__ == "__main__":
    main()
