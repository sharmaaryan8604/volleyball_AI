"""
pre_train.py — Run this ONCE locally (or in your Render build command)
to train all models and save them to models/ with joblib.

Usage:
    python pre_train.py

After running, commit the models/ directory to git OR upload it to
Render as a persistent disk. The API will then load instantly
from disk instead of retraining on every cold start.

Render build command (recommended):
    pip install -r api/requirements.txt && python pre_train.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, clean_dataset
from src.spatial_features import add_spatial_features
from src.ml_model import train_ml_model
from src.markov_model import build_transition_matrix

MODEL_DIR = Path("models")

ALL_FEATURES = [
    "receive_location","digger_location","pass_land_location",
    "hitter_location","set_location","pass_rating","set_type",
    "hit_type","serve_type","num_blockers","block_touch","rally","round",
    "hx","hy","dist_net","dist_center",
    "attack_dx","attack_dy","attack_distance","attack_angle",
    "traj_dx","traj_dy","traj_distance","traj_angle",
    "cross_court","line_attack","back_row","deep_attack","short_attack",
    "attack_pressure","quality_pressure",
]


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ────────────────────────────────────────────────────
    print("Loading data…")
    train, test = load_data("data/training data.csv", "data/testing data.csv")
    train = clean_dataset(train)
    test  = clean_dataset(test)
    train = add_spatial_features(train)
    test  = add_spatial_features(test)

    features = [f for f in ALL_FEATURES if f in train.columns]
    print(f"Features: {len(features)}")

    # ── Rally-based split ────────────────────────────────────────────────────
    X = train[features]
    y = train["hit_land_location"] - 1

    unique_rallies = train["rally"].unique()
    train_rallies, val_rallies = train_test_split(
        unique_rallies, test_size=0.2, random_state=42
    )
    mask = train["rally"].isin(train_rallies)
    print(f"Train rows: {mask.sum():,}  |  Val rows: {(~mask).sum():,}")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining XGBoost + LightGBM…")
    t1 = time.time()
    xgb_model, lgb_model = train_ml_model(X[mask], y[mask])
    print(f"  done in {time.time()-t1:.1f}s")

    print("Building Markov transition matrix…")
    t2 = time.time()
    markov_matrix = build_transition_matrix(train[mask])
    print(f"  done in {time.time()-t2:.1f}s  |  {len(markov_matrix):,} states")

    zone_counts = train.loc[mask, "hit_land_location"].value_counts(normalize=True)
    zone_prior  = {int(k) - 1: float(v) for k, v in zone_counts.items()}

    # ── Quick validation ─────────────────────────────────────────────────────
    from sklearn.metrics import top_k_accuracy_score

    X_val = X[~mask]
    y_val = y[~mask]
    xgb_p = xgb_model.predict_proba(X_val)
    lgb_p = lgb_model.predict_proba(X_val)
    probs = 0.5 * xgb_p + 0.5 * lgb_p

    top1 = top_k_accuracy_score(y_val, probs, k=1)
    top3 = top_k_accuracy_score(y_val, probs, k=3)
    print(f"\nValidation  Top-1: {top1*100:.2f}%  |  Top-3: {top3*100:.2f}%")

    # ── Serialize ────────────────────────────────────────────────────────────
    print("\nSaving models…")
    joblib.dump(xgb_model,     MODEL_DIR / "xgb_model.joblib",     compress=3)
    joblib.dump(lgb_model,     MODEL_DIR / "lgb_model.joblib",     compress=3)
    joblib.dump(markov_matrix, MODEL_DIR / "markov_matrix.joblib", compress=3)
    joblib.dump(zone_prior,    MODEL_DIR / "zone_prior.joblib",    compress=3)
    joblib.dump(features,      MODEL_DIR / "features.joblib",      compress=3)

    # Print file sizes
    total = 0
    for f in MODEL_DIR.glob("*.joblib"):
        size = f.stat().st_size / 1024
        total += size
        print(f"  {f.name:<30} {size:>7.1f} KB")
    print(f"  {'TOTAL':<30} {total:>7.1f} KB")

   

if __name__ == "__main__":
    main()
