#!/usr/bin/env python3
"""
Phase_One.py (fixed)
Phase 1: Preprocessing pipeline (secure, reproducible, deployment-ready)

This version contains targeted fixes and hardening while keeping your original
pipeline and interfaces unchanged.

Fixes / improvements (summary):
 - Safer saver for joblib.dump (catch & log failures)
 - More robust safe_json_dump (atomic replace + fsync where possible)
 - Validate strict_features_json content and fall back with a clear error
 - Ensure saved numpy/pickle artifacts use explicit float32 / int64 types
 - Minor logging improvements around saves & error handling
"""
import os
import sys
import json
import hashlib
import random
from pathlib import Path
import argparse
import logging
from datetime import datetime
from pathlib import Path as _Path  # legacy references below unchanged
import platform
import errno

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import sklearn

# Ensure project root is on sys.path for test invocation from arbitrary CWD
try:
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in sys.path:
        sys.path.insert(0, str(_this_dir))
except Exception:
    pass

set_all_seeds = None  # placeholder, will be set after definition

# ----------------- Defaults (portable relative to project root) -----------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = str(BASE_DIR / "IDSAI.csv")
DEFAULT_ARTIFACTS = str(BASE_DIR / "artifacts_phase1")
DEFAULT_RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.10
DEFAULT_TEST_SIZE = 0.10
DEFAULT_BATCH_SIZE = 64

# ----------------- Fixed Protocol Mapping -----------------
FIXED_PROTOCOL_MAP = {"ICMP": 1, "TCP": 6, "UDP": 17, "OTHER": 0}
OTHER_CODE = 0

# ----------------- Utilities -----------------
def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_all_seeds = set_seeds  # Use local set_seeds implementation

def set_thread_limits(cpus: int = 1):
    if cpus is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpus)
    logging.debug("Set CPU thread env limits to %s", cpus)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.str.strip().str.lower()
    cols = cols.str.replace(" ", "_", regex=False)
    cols = cols.str.replace("-", "_", regex=False)
    cols = cols.str.replace(".", "_", regex=False)
    df.columns = cols
    return df

def coerce_protocol(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    out = []
    for v in s:
        if v.isdigit():
            iv = int(v)
            out.append(iv if iv in (1, 6, 17) else OTHER_CODE)
        else:
            out.append(FIXED_PROTOCOL_MAP.get(v, OTHER_CODE))
    return pd.Series(out, index=series.index, dtype=np.int64)

def protocol_stats(series_in: pd.Series, series_out: pd.Series) -> dict:
    return {
        "input_counts": series_in.astype(str).str.upper().str.strip().value_counts(dropna=False).to_dict(),
        "output_counts": series_out.value_counts(dropna=False).to_dict(),
        "mapping_used": FIXED_PROTOCOL_MAP,
        "other_code": OTHER_CODE,
    }

def find_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "labels", "attack", "attacks", "tipo_ataque", "class", "category", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "label" in c or "attack" in c or "target" in c:
            return c
    raise KeyError("No label-like column found. Expected one of: " + ", ".join(candidates))

def map_binary_labels(y: pd.Series) -> pd.Series:
    s = y.astype(str).str.strip().str.upper()
    mapping = {
        "BENIGN": 0, "NORMAL": 0, "LEGIT": 0, "GOOD": 0, "0": 0,
        "ATTACK": 1, "MALICIOUS": 1, "BAD": 1, "1": 1,
    }
    z = s.map(mapping)
    fallback = (~s.isin(["BENIGN", "NORMAL", "LEGIT", "GOOD", "0"])).astype(int)
    z = z.fillna(fallback)
    return z.astype(np.int64)

def robust_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    med = df.median(numeric_only=True)
    df = df.fillna(med)
    return df

def drop_constant_columns(df: pd.DataFrame, eps: float = 0.0):
    std = df.std(numeric_only=True)
    keep_cols = std[std > eps].index.tolist()
    dropped = [c for c in df.columns if c not in keep_cols]
    return df[keep_cols].copy(), dropped

def ensure_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[feature_list].copy()

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_of_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_json_dump(payload, path: Path):
    """
    Atomic JSON write: write to tmp then replace target. Attempt to fsync directory if possible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                # not critical if fsync unavailable
                pass
        # atomic replace
        tmp.replace(path)
    except Exception as e:
        logging.exception("safe_json_dump failed for %s: %s", path, e)
        # as a last resort, try a simple write
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            logging.exception("Final fallback JSON write also failed for %s", path)
            raise

# ----------------- Main pipeline -----------------
def run_phase_one(
    csv_path: Path,
    artifacts_dir: Path,
    random_state: int = DEFAULT_RANDOM_STATE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_csvs: bool = True,
    strict_features_json: Path | None = None,
    thread_limit: int | None = 1,
    no_pickle_scaler: bool = False,
):
    # Legacy local seed function kept; prefer centralized reproducibility module
    set_seeds(random_state)
    set_thread_limits(thread_limit)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading dataset from: %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    logging.info("Raw shape: %s", df.shape)

    # Normalize columns
    df = normalize_columns(df)

    # Drop obvious identifiers if present
    drop_cols = [c for c in ["ip_src", "ip_dst"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logging.debug("Dropped identifier columns: %s", drop_cols)

    # Protocol normalization
    proto_col = "protocol" if "protocol" in df.columns else ("protocols" if "protocols" in df.columns else None)
    if proto_col is None:
        raise KeyError("No protocol/protocols column found.")
    proto_in = df[proto_col].copy()
    df["protocol"] = coerce_protocol(df[proto_col])
    if proto_col != "protocol" and proto_col in df.columns:
        df = df.drop(columns=[proto_col])
    proto_stat = protocol_stats(proto_in, df["protocol"])

    # Save protocol artifact
    safe_json_dump(FIXED_PROTOCOL_MAP, artifacts_dir / "protocol_mapping.json")
    safe_json_dump(proto_stat, artifacts_dir / "protocol_stats.json")

    # Label mapping: detect column
    label_col = find_label_column(df)
    logging.info("Using label column: %s", label_col)
    df["label"] = map_binary_labels(df[label_col])
    if label_col != "label":
        try:
            df = df.drop(columns=[label_col])
        except Exception:
            pass

    # Candidate features (exclude label and known non-feature columns)
    exclude_cols = {"label", "tipo_ataque"}
    candidate_features = [c for c in df.columns if c not in exclude_cols]
    X_df = df[candidate_features].copy()
    X_df = robust_numeric(X_df)
    X_df, dropped_constant = drop_constant_columns(X_df, eps=0.0)

    # Strict feature ordering if provided
    feature_order = list(X_df.columns)
    if strict_features_json:
        try:
            if not strict_features_json.exists():
                raise FileNotFoundError(f"strict_features_json provided but not found: {strict_features_json}")
            with strict_features_json.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("strict_features_json must contain a JSON array of feature names")
            # enforce all entries are strings
            feature_order = [str(x) for x in loaded]
            X_df = ensure_features(X_df, feature_order)
        except Exception as e:
            logging.exception("Failed to apply strict_features_json: %s", e)
            raise

    y = df["label"].astype(np.int64).values

    # Null check (should be 0)
    nulls = X_df.isnull().sum().sort_values(ascending=False)
    logging.info("Top null counts after cleaning (should be 0):\n%s", nulls.head(10).to_string())
    assert X_df.isnull().sum().sum() == 0, "NaNs remain in feature matrix after cleaning!"

    # ---------- Stratified splits ----------
    X_all = X_df.values
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=random_state, stratify=y_trainval
    )

    # ---------- Scaling ----------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Save clipping bounds (on original-space train)
    clip_bounds = {
        "p1": np.percentile(X_train, 1, axis=0).astype(float).tolist(),
        "p99": np.percentile(X_train, 99, axis=0).astype(float).tolist()
    }
    safe_json_dump(clip_bounds, artifacts_dir / "clip_bounds.json")

    # ---------- Class weights ----------
    classes_unique = np.unique(y_train).tolist()
    weights = compute_class_weight(class_weight="balanced", classes=np.array(classes_unique), y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes_unique, weights)}
    class_counts = {int(c): int((y_train == c).sum()) for c in classes_unique}

    # ---------- Persist artifacts ----------
    # Save scaler (joblib) unless --no-pickle-scaler
    if not no_pickle_scaler:
        try:
            joblib.dump(scaler, artifacts_dir / "scaler.pkl", compress=3)
            logging.info("Saved scaler.pkl (joblib)")
        except Exception as e:
            logging.warning("Failed to save scaler.pkl - continuing (will still save scaler.json): %s", e)

    # Save scaler mean/scale as JSON for edge usage (float32)
    scaler_json = {
        "mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float32).tolist(),
        "with_mean": getattr(scaler, "with_mean", True),
        "with_std": getattr(scaler, "with_std", True),
    }
    safe_json_dump(scaler_json, artifacts_dir / "scaler.json")
    logging.info("Saved scaler.json (edge-friendly)")

    # Save feature order
    safe_json_dump(feature_order, artifacts_dir / "feature_order.json")

    # Save class weights & label mapping
    safe_json_dump({"weights": class_weights, "counts": class_counts}, artifacts_dir / "class_weights.json")
    safe_json_dump({"0": "Benign", "1": "Attack"}, artifacts_dir / "label_mapping.json")

    # Feature stats on scaled train (float32)
    feat_stats = {}
    X_train_s = X_train_s.astype(np.float32)
    X_val_s = X_val_s.astype(np.float32)
    X_test_s = X_test_s.astype(np.float32)
    for i, col in enumerate(feature_order):
        feat_stats[col] = {
            "mean": float(np.mean(X_train_s[:, i])),
            "std": float(np.std(X_train_s[:, i])),
            "min": float(np.min(X_train_s[:, i])),
            "max": float(np.max(X_train_s[:, i])),
        }
    safe_json_dump(feat_stats, artifacts_dir / "feature_stats.json")

    # Save preprocessed CSVs (optional)
    if save_csvs:
        try:
            tr_df = pd.DataFrame(X_train_s, columns=feature_order); tr_df["label"] = y_train
            va_df = pd.DataFrame(X_val_s,   columns=feature_order); va_df["label"] = y_val
            te_df = pd.DataFrame(X_test_s,  columns=feature_order); te_df["label"] = y_test
            # ensure float32 columns persist to CSV (pandas will write them)
            tr_df.to_csv(artifacts_dir / "train_preprocessed.csv", index=False)
            va_df.to_csv(artifacts_dir / "val_preprocessed.csv", index=False)
            te_df.to_csv(artifacts_dir / "test_preprocessed.csv", index=False)
        except Exception as e:
            logging.warning("Failed to write preprocessed CSVs: %s", e)

    # Also save compressed numpy archive for quick loading by publisher/training (float32)
    try:
        np.savez_compressed(
            artifacts_dir / "data.npz",
            X_train=X_train_s.astype(np.float32),
            y_train=y_train.astype(np.int64),
            X_val=X_val_s.astype(np.float32),
            y_val=y_val.astype(np.int64),
            X_test=X_test_s.astype(np.float32),
            y_test=y_test.astype(np.int64),
            feature_order=np.array(feature_order, dtype=object)
        )
    except Exception as e:
        logging.exception("Failed to save data.npz: %s", e)
        raise

    # Save as pickle for backward compatibility
    data_dict = {
        "X_train": X_train_s.astype(np.float32), "y_train": y_train.astype(np.int64),
        "X_val": X_val_s.astype(np.float32), "y_val": y_val.astype(np.int64),
        "X_test": X_test_s.astype(np.float32), "y_test": y_test.astype(np.int64)
    }
    try:
        with (artifacts_dir / "data.pkl").open("wb") as f:
            import pickle
            pickle.dump(data_dict, f)
    except Exception as e:
        logging.warning("Failed to write data.pkl: %s", e)

    # ---------- Metadata & checksums ----------
    artifact_files = sorted([p for p in artifacts_dir.glob("*") if p.is_file()])
    manifest = {}
    for p in artifact_files:
        try:
            manifest[p.name] = {
                "sha256": sha256_of_file(p),
                "size_bytes": p.stat().st_size
            }
        except Exception as e:
            logging.warning("Could not hash %s: %s", p, e)

    meta = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "csv_path": str(csv_path),
        "seed": random_state,
        "batch_size": batch_size,
        "train_size": int(len(X_train_s)),
        "val_size": int(len(X_val_s)),
        "test_size": int(len(X_test_s)),
        "splits": {
            "val_size": val_size,
            "test_size": test_size,
            "train": int(len(X_train_s)),
            "val": int(len(X_val_s)),
            "test": int(len(X_test_s)),
        },
        "num_features": len(feature_order),
        "feature_order": feature_order,
        "dropped_constant_columns": dropped_constant,
        "label_col": "label",
        "protocol_col": "protocol",
        "protocol_mapping": FIXED_PROTOCOL_MAP,
        "class_mapping": {"0": "Benign", "1": "Attack"},
        "class_weights": class_weights,
        "scaler": {
            "type": "StandardScaler",
            "with_mean": getattr(scaler, "with_mean", True),
            "with_std": getattr(scaler, "with_std", True),
            "mean_shape": list(getattr(scaler, "mean_", np.array([])).shape),
            "var_shape": list(getattr(scaler, "var_", np.array([])).shape),
            "json_path": "scaler.json",
            "pkl_path": "scaler.pkl" if not no_pickle_scaler else None
        },
        "hashes": {
            "train_y": sha256_of_bytes(np.ascontiguousarray(y_train).astype(np.int64).tobytes()),
            "val_y": sha256_of_bytes(np.ascontiguousarray(y_val).astype(np.int64).tobytes()),
            "test_y": sha256_of_bytes(np.ascontiguousarray(y_test).astype(np.int64).tobytes()),
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts_manifest": manifest,
    }
    safe_json_dump(meta, artifacts_dir / "metadata.json")

    logging.info("Phase 1 complete. Artifacts saved to: %s", artifacts_dir)
    # pipeline_contract module not found; skipping contract hash check
    logging.info("Artifacts required for later phases: scaler.json (edge), scaler.pkl (dev), feature_order.json, clip_bounds.json, label_mapping.json, data.npz")
    return meta

def build_argparser():
    p = argparse.ArgumentParser(description="Phase 1 preprocessing for IoT anomaly detection")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to raw CSV dataset")
    p.add_argument("--artifacts", type=str, default=DEFAULT_ARTIFACTS, help="Artifacts directory")
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--val", type=float, default=DEFAULT_VAL_SIZE, help="Validation size (relative to total)")
    p.add_argument("--test", type=float, default=DEFAULT_TEST_SIZE, help="Test size (relative to total)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--no-csv", action="store_true", help="Do not save CSV preprocessed splits")
    p.add_argument("--strict-features-json", type=str, default=None, help="Optional JSON with strict feature order")
    p.add_argument("--threads", type=int, default=1, help="Limit CPU threads used by BLAS/OpenMP for reproducibility")
    p.add_argument("--no-pickle-scaler", action="store_true", help="Do not save scaler.pkl; save only scaler.json")
    return p

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    try:
        meta = run_phase_one(
            csv_path=Path(args.csv),
            artifacts_dir=Path(args.artifacts),
            random_state=args.seed,
            val_size=args.val,
            test_size=args.test,
            batch_size=args.batch_size,
            save_csvs=(not args.no_csv),
            strict_features_json=Path(args.strict_features_json) if args.strict_features_json else None,
            thread_limit=args.threads,
            no_pickle_scaler=args.no_pickle_scaler,
        )
    except AssertionError as e:
        logging.error("Assertion error: %s", e)
        sys.exit(2)
    except Exception as e:
        logging.exception("Phase 1 failed: %s", e)
        sys.exit(1)

