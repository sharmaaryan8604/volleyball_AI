"""
Volleyball AI — FastAPI Backend
Wraps the XGBoost + LightGBM + Markov hybrid model from src/

Model caching strategy:
  - On first startup, trains all models from the CSV data and saves to models/
  - On every subsequent startup, loads instantly from the saved files (~1-2s)
  - To force a retrain: delete models/ directory or set RETRAIN=1 env var
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
from pathlib import Path

# ── Model cache directory ────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

CACHE_FILES = {
    "xgb":     MODEL_DIR / "xgb_model.joblib",
    "lgb":     MODEL_DIR / "lgb_model.joblib",
    "markov":  MODEL_DIR / "markov_matrix.joblib",
    "prior":   MODEL_DIR / "zone_prior.joblib",
    "features": MODEL_DIR / "features.joblib",
}

# ── In-memory globals (populated once per process) ───────────────────────────
_xgb_model     = None
_lgb_model     = None
_markov_matrix = None
_zone_prior    = None
_features      = None


def _cache_is_valid() -> bool:
    """All cache files exist AND RETRAIN env var is not set."""
    if os.getenv("RETRAIN", "0") == "1":
        return False
    return all(p.exists() for p in CACHE_FILES.values())


def _save_cache(xgb, lgb, markov, prior, features):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb,      CACHE_FILES["xgb"],      compress=3)
    joblib.dump(lgb,      CACHE_FILES["lgb"],      compress=3)
    joblib.dump(markov,   CACHE_FILES["markov"],   compress=3)
    joblib.dump(prior,    CACHE_FILES["prior"],    compress=3)
    joblib.dump(features, CACHE_FILES["features"], compress=3)
    print(f"[cache] Models saved to {MODEL_DIR}/")


def _load_cache():
    xgb      = joblib.load(CACHE_FILES["xgb"])
    lgb      = joblib.load(CACHE_FILES["lgb"])
    markov   = joblib.load(CACHE_FILES["markov"])
    prior    = joblib.load(CACHE_FILES["prior"])
    features = joblib.load(CACHE_FILES["features"])
    print(f"[cache] Loaded models from {MODEL_DIR}/ — skipping training")
    return xgb, lgb, markov, prior, features


def _train_and_cache():
    """Full training pipeline. Called only when no valid cache exists."""
    from src.preprocessing import load_data, clean_dataset
    from src.spatial_features import add_spatial_features
    from src.ml_model import train_ml_model
    from src.markov_model import build_transition_matrix
    from sklearn.model_selection import train_test_split

    print("[train] No cache found — training from scratch…")

    train, _ = load_data("data/training data.csv", "data/testing data.csv")
    train = clean_dataset(train)
    train = add_spatial_features(train)

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
    features = [f for f in ALL_FEATURES if f in train.columns]
    print(f"[train] Using {len(features)} features")

    X = train[features]
    y = train["hit_land_location"] - 1

    unique_rallies = train["rally"].unique()
    train_rallies, _ = train_test_split(unique_rallies, test_size=0.2, random_state=42)
    mask = train["rally"].isin(train_rallies)

    print(f"[train] Train rows: {mask.sum():,}  |  Val rows: {(~mask).sum():,}")

    xgb_model, lgb_model = train_ml_model(X[mask], y[mask])
    print("[train] XGBoost + LightGBM trained")

    markov_matrix = build_transition_matrix(train[mask])
    print(f"[train] Markov matrix: {len(markov_matrix):,} states")

    zone_counts = train.loc[mask, "hit_land_location"].value_counts(normalize=True)
    zone_prior  = {int(k) - 1: float(v) for k, v in zone_counts.items()}

    _save_cache(xgb_model, lgb_model, markov_matrix, zone_prior, features)
    return xgb_model, lgb_model, markov_matrix, zone_prior, features


def get_models():
    """
    Returns (xgb, lgb, markov, zone_prior, features).

    Flow:
      1. Already loaded in memory → return immediately (sub-millisecond)
      2. Cache files exist on disk → joblib.load (~1-2s)
      3. No cache → full training pipeline + save to disk (~60-90s once)
    """
    global _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features

    # Already in memory
    if _xgb_model is not None:
        return _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features

    try:
        if _cache_is_valid():
            _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features = _load_cache()
        else:
            _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features = _train_and_cache()
    except Exception as e:
        raise RuntimeError(f"Model initialisation failed: {e}")

    return _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Volleyball AI API",
    description="XGBoost + LightGBM + Markov chain hybrid for hit landing prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    receive_location:    Optional[float] = None
    digger_location:     Optional[float] = None
    pass_land_location:  Optional[float] = None
    hitter_location:     float = Field(..., ge=8, le=15, description="Hitter zone 8-15")
    set_location:        float = Field(..., ge=1, le=8,  description="Encoded set type: outside=1, oppo=2, quick=3, bic=4, dump=5, d-ball=6, in=7, blocked=8")
    pass_rating:         int   = Field(..., ge=0, le=1,  description="Pass quality: 0=bad, 1=good")
    set_type:            Optional[float] = None
    hit_type:            Optional[float] = None
    serve_type:          Optional[float] = None
    num_blockers:        Optional[float] = None
    block_touch:         Optional[int]   = None
    rally:               Optional[float] = None
    round:               Optional[float] = None

class ZoneProbability(BaseModel):
    zone: int
    probability: float
    label: str

class PredictResponse(BaseModel):
    top_zones:       List[ZoneProbability]
    top1_zone:       int
    top3_zones:      List[int]
    top5_zones:      List[int]
    markov_hit:      bool
    all_probs:       List[float]

class SimulateRequest(BaseModel):
    hitter_zone:  int   = Field(..., ge=8, le=15)
    pass_rating:  int   = Field(..., ge=0, le=1)
    set_loc:      int   = Field(..., ge=1, le=8)
    n:            int   = Field(1000, ge=100, le=10000)

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    cache_exists: bool
    model_dir:    str

class MarkovStatesResponse(BaseModel):
    states:     int
    matrix_keys: List[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

ZONE_LABELS = {
    1: "Zone 1",  2: "Zone 2",  3: "Zone 3",  4: "Zone 4",  5: "Zone 5",
    6: "Zone 6",  7: "Zone 7",  8: "Zone 8",  9: "Zone 9", 10: "Zone 10",
    11: "Zone 11", 12: "Zone 12", 13: "Zone 13", 14: "Zone 14", 15: "Zone 15",
    16: "Zone 16", 17: "Zone 17", 18: "Zone 18", 19: "Zone 19", 20: "Zone 20",
    21: "Zone 21", 22: "Zone 22", 23: "Zone 23", 24: "Zone 24", 25: "Zone 25",
}

def row_to_df(req: PredictRequest, features: List[str]) -> pd.DataFrame:
    data = req.dict()
    # Fill any spatial features with reasonable defaults (0)
    row = {f: data.get(f, 0) for f in features}
    return pd.DataFrame([row])


def blend_probs(xgb_p, lgb_p, markov_matrix, zone_prior, req: PredictRequest):
    from src.markov_model import lookup_transition

    probs = 0.5 * xgb_p + 0.5 * lgb_p
    markov_hit = False

    markov_probs = lookup_transition(
        markov_matrix, req.pass_rating, req.set_location, req.hitter_location
    )
    if markov_probs:
        markov_hit = True
        for zone, p in markov_probs.items():
            zone_idx = int(zone) - 1
            if 0 <= zone_idx < len(probs):
                probs[zone_idx] = 0.70 * probs[zone_idx] + 0.30 * p

    prior_weight = 0.10
    for zone, p in zone_prior.items():
        if 0 <= zone < len(probs):
            probs[zone] = (1 - prior_weight) * probs[zone] + prior_weight * p

    return probs, markov_hit


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "model_loaded": _xgb_model is not None,
        "cache_exists": _cache_is_valid(),
        "model_dir": str(MODEL_DIR.resolve()),
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": _xgb_model is not None,
        "cache_exists": _cache_is_valid(),
        "model_dir": str(MODEL_DIR.resolve()),
    }


@app.post("/admin/retrain", tags=["admin"])
def retrain():
    """
    Force a full retrain and overwrite the cached models.
    Useful after updating training data without redeploying.
    Protect this endpoint with an API key in production.
    """
    global _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features
    # Clear in-memory cache so _train_and_cache() is called
    _xgb_model = _lgb_model = _markov_matrix = _zone_prior = _features = None
    # Delete existing files
    for p in CACHE_FILES.values():
        p.unlink(missing_ok=True)
    try:
        _xgb_model, _lgb_model, _markov_matrix, _zone_prior, _features = _train_and_cache()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "retrained", "model_dir": str(MODEL_DIR.resolve())}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        xgb_model, lgb_model, markov_matrix, zone_prior, features = get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    df = row_to_df(req, features)

    try:
        xgb_p = xgb_model.predict_proba(df)[0]
        lgb_p = lgb_model.predict_proba(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    probs, markov_hit = blend_probs(xgb_p, lgb_p, markov_matrix, zone_prior, req)
    sorted_idx = np.argsort(probs)[::-1]

    top_zones = [
        ZoneProbability(
            zone=int(sorted_idx[i]) + 1,
            probability=float(probs[sorted_idx[i]]),
            label=ZONE_LABELS.get(int(sorted_idx[i]) + 1, f"Zone {int(sorted_idx[i]) + 1}")
        )
        for i in range(min(10, len(sorted_idx)))
    ]

    return PredictResponse(
        top_zones=top_zones,
        top1_zone=int(sorted_idx[0]) + 1,
        top3_zones=[int(sorted_idx[i]) + 1 for i in range(3)],
        top5_zones=[int(sorted_idx[i]) + 1 for i in range(5)],
        markov_hit=markov_hit,
        all_probs=[float(p) for p in probs],
    )


@app.post("/simulate")
def simulate(req: SimulateRequest):
    try:
        _, _, markov_matrix, _, _ = get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        from src.simulation import simulate_landing_distribution
        dist = simulate_landing_distribution(
            markov_matrix,
            hitter_zone=req.hitter_zone,
            pass_rating=req.pass_rating,
            set_loc=req.set_loc,
            n=req.n,
        )
        return {
            "distribution": {str(k): int(v) for k, v in dist.items()},
            "n": req.n,
            "hitter_zone": req.hitter_zone,
            "pass_rating": req.pass_rating,
            "set_loc": req.set_loc,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/markov/info", response_model=MarkovStatesResponse)
def markov_info():
    try:
        _, _, markov_matrix, _, _ = get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    keys = [str(k) for k in list(markov_matrix.keys())[:20]]
    return {"states": len(markov_matrix), "matrix_keys": keys}


@app.get("/zones/prior")
def zone_prior_endpoint():
    try:
        _, _, _, zone_prior, _ = get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"zone_prior": {str(int(k)+1): float(v) for k, v in zone_prior.items()}}
