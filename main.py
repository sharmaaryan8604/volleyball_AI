import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, clean_dataset
from src.spatial_features import add_spatial_features
from src.ml_model import train_ml_model
from src.markov_model import build_transition_matrix, lookup_transition


# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

print("Loading data...")

train, test = load_data(
    "data/training data.csv",
    "data/testing data.csv"
)

train = clean_dataset(train)
test  = clean_dataset(test)

train = add_spatial_features(train)
test  = add_spatial_features(test)


# -----------------------------------------------------
# Feature Selection
# -----------------------------------------------------

features = [
    "receive_location",
    "digger_location",
    "pass_land_location",
    "hitter_location",
    "set_location",
    "pass_rating",       # now 0/1 integer (was "in"/"out" string)
    "set_type",
    "hit_type",
    "serve_type",
    "num_blockers",
    "block_touch",       # now 0/1 integer (was "yes"/"no" string)
    "rally",
    "round",

    # spatial features (now correct — built from integer zones)
    "hx",
    "hy",
    "dist_net",
    "dist_center",

    # attack vectors
    "attack_dx",
    "attack_dy",
    "attack_distance",
    "attack_angle",

    # trajectory
    "traj_dx",
    "traj_dy",
    "traj_distance",
    "traj_angle",

    # indicators
    "cross_court",
    "line_attack",
    "back_row",
    "deep_attack",
    "short_attack",

    # engineered
    "attack_pressure",
    "quality_pressure"
]

features = [f for f in features if f in train.columns]
print("\nFeature count:", len(features))


# -----------------------------------------------------
# Prepare Dataset
# -----------------------------------------------------

X = train[features]
y = train["hit_land_location"] - 1      # XGBoost needs 0-indexed labels (0-24)


# -----------------------------------------------------
# Train / Validation Split
# FIX: split by rally ID, not randomly, to prevent
# data leakage between consecutive rallies
# -----------------------------------------------------

unique_rallies = train["rally"].unique()

train_rallies, val_rallies = train_test_split(
    unique_rallies,
    test_size=0.2,
    random_state=42
)

train_mask = train["rally"].isin(train_rallies)
val_mask   = train["rally"].isin(val_rallies)

X_train, X_test = X[train_mask], X[val_mask]
y_train, y_test = y[train_mask], y[val_mask]

print(f"\nTrain rows: {len(X_train):,}  |  Val rows: {len(X_test):,}")
print(f"Rally-based split — no leakage between train and val.")


# -----------------------------------------------------
# Baselines (so we know what "good" looks like)
# -----------------------------------------------------

n_classes = len(np.unique(y))

# Random baseline
random_probs   = np.ones((len(y_test), n_classes)) / n_classes
random_top1    = top_k_accuracy_score(y_test, random_probs, k=1)

# Majority-class baseline
majority_class = np.bincount(y_train).argmax()
majority_probs = np.zeros((len(y_test), n_classes))
majority_probs[:, majority_class] = 1.0
majority_top1  = top_k_accuracy_score(y_test, majority_probs, k=1)

print(f"\nBaseline — Random:         Top-1 = {random_top1*100:.2f}%")
print(f"Baseline — Majority class: Top-1 = {majority_top1*100:.2f}%")
print(f"Your model must beat {majority_top1*100:.2f}% to be useful.\n")


# -----------------------------------------------------
# Train ML Models
# -----------------------------------------------------

print("Training ML models...")
xgb_model, lgb_model = train_ml_model(X_train, y_train)
print("Models trained.")


# -----------------------------------------------------
# Build Markov Sequence Model
# FIX: uses encoded integer keys (pass_rating=0/1,
# set_location=1-8, hitter_location=1-15) instead of
# strings that crashed int() conversion silently
# -----------------------------------------------------

print("\nBuilding Markov model...")
markov_matrix = build_transition_matrix(train[train_mask])

# Sanity check — how many states were learned?
print(f"Markov states learned: {len(markov_matrix):,}")


# -----------------------------------------------------
# Zone Priors
# -----------------------------------------------------

zone_counts = train.loc[train_mask, "hit_land_location"].value_counts(normalize=True)
zone_prior  = {int(k) - 1: float(v) for k, v in zone_counts.items()}


# -----------------------------------------------------
# Ensemble Prediction
# -----------------------------------------------------

print("\nEvaluating model...")

xgb_probs = xgb_model.predict_proba(X_test)
lgb_probs = lgb_model.predict_proba(X_test)

probs = 0.5 * xgb_probs + 0.5 * lgb_probs


# -----------------------------------------------------
# Hybrid Blending (ML + Markov + Priors)
# FIX: uses lookup_transition() which handles encoded
# integer keys correctly and falls back gracefully
# -----------------------------------------------------

markov_hits  = 0
markov_total = 0

for i, row in X_test.iterrows():

    idx = X_test.index.get_loc(i)

    pass_rating = row.get("pass_rating")
    set_loc     = row.get("set_location")
    hitter      = row.get("hitter_location")

    if pd.isna(pass_rating) or pd.isna(set_loc) or pd.isna(hitter):
        continue

    markov_total += 1

    markov_probs = lookup_transition(markov_matrix, pass_rating, set_loc, hitter)

    if markov_probs:
        markov_hits += 1
        for zone, p in markov_probs.items():
            zone_idx = int(zone) - 1
            if 0 <= zone_idx < probs.shape[1]:
                probs[idx, zone_idx] = (
                    0.70 * probs[idx, zone_idx] +
                    0.30 * p
                )

    # Zone prior blending
    prior_weight = 0.10
    for zone, p in zone_prior.items():
        if 0 <= zone < probs.shape[1]:
            probs[idx, zone] = (
                (1 - prior_weight) * probs[idx, zone] +
                prior_weight * p
            )

print(f"Markov lookup hit rate: {markov_hits}/{markov_total} "
      f"({markov_hits/max(markov_total,1)*100:.1f}%) — "
      f"was 0% before fix")


# -----------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------

top1 = top_k_accuracy_score(y_test, probs, k=1)
top3 = top_k_accuracy_score(y_test, probs, k=3)
top5 = top_k_accuracy_score(y_test, probs, k=5)

ranks = []
for i in range(len(y_test)):
    sorted_idx = np.argsort(probs[i])[::-1]
    true_zone  = int(y_test.iloc[i])
    match      = np.where(sorted_idx == true_zone)[0]
    ranks.append(1 / (match[0] + 1) if len(match) > 0 else 0)

mrr = np.mean(ranks)


# -----------------------------------------------------
# Results vs Baselines
# -----------------------------------------------------

print("\n===== FINAL RESULTS =====\n")
print(f"Random   baseline Top-1 : {random_top1*100:.2f}%")
print(f"Majority baseline Top-1 : {majority_top1*100:.2f}%")
print(f"-------------------------------")
print(f"Your model  Top-1  : {round(top1 * 100, 2)}%")
print(f"Your model  Top-3  : {round(top3 * 100, 2)}%")
print(f"Your model  Top-5  : {round(top5 * 100, 2)}%")
print(f"Your model  MRR    : {round(mrr, 4)}")
print(f"\nLift over majority: +{(top1 - majority_top1)*100:.2f}pp")


# -----------------------------------------------------
# Attack Map
# -----------------------------------------------------

from src.attack_map import plot_attack_map
plot_attack_map(markov_matrix)


# ─────────────────────────────────────────────────────────────
# Extended Evaluation
# ─────────────────────────────────────────────────────────────

from src.evaluation import (
    full_evaluation_report,
    per_zone_accuracy,
    confusion_analysis,
    zone_coverage_report,
    mrr_by_hitter_zone,
    plot_evaluation_dashboard,
)

print("\n===== EXTENDED EVALUATION =====\n")

metrics = full_evaluation_report(y_test, probs, y_train=y_train,
                                  X_test=X_test, label="XGB+LGB+Markov Hybrid")

print("\n── Top-k Coverage ──")
print(zone_coverage_report(y_test, probs).to_string())

print("\n── Per-Zone Top-1 Accuracy (hardest zones first) ──")
print(per_zone_accuracy(y_test, probs, k=1).head(10).to_string())

print("\n── Top-10 Most Common Errors ──")
print(confusion_analysis(y_test, probs, top_n=10).to_string(index=False))

print("\n── MRR by Hitter Zone ──")
print(mrr_by_hitter_zone(y_test, probs, X_test).to_string())

plot_evaluation_dashboard(y_test, probs, X_test=X_test,
                           title="Volleyball AI — Full Evaluation Dashboard")


# ─────────────────────────────────────────────────────────────
# Simulations
# ─────────────────────────────────────────────────────────────

from src.simulation import (
    simulate_landing_distribution,
    simulate_scenario,
    simulate_rally_sequence,
    simulate_pass_quality_effect,
    simulate_win_probability,
)

print("\n===== SIMULATIONS =====\n")

# 1. Landing distribution from zone 11 (back-left, good pass, outside set)
print("── Sim 1: Landing distribution — Zone 11, good pass, outside set ──")
dist = simulate_landing_distribution(markov_matrix, hitter_zone=11,
                                      pass_rating=1, set_loc=1, n=10_000)
for zone, count in list(dist.items())[:8]:
    bar = "█" * int(count / 50)
    print(f"  Zone {zone:2d}: {count:5d}  {bar}")

# 2. Scenario comparison: good vs bad pass vs quick set
print("\n── Sim 2: Scenario comparison — Zone 11 ──")
scenarios = [
    {"label": "Good pass + Outside", "pass_rating": 1, "set_loc": 1},
    {"label": "Bad pass  + Outside", "pass_rating": 0, "set_loc": 1},
    {"label": "Good pass + Quick",   "pass_rating": 1, "set_loc": 3},
    {"label": "Good pass + Oppo",    "pass_rating": 1, "set_loc": 2},
]
sc_df = simulate_scenario(markov_matrix, hitter_zone=11,
                           scenarios=scenarios, n=10_000)
print(sc_df.head(8).to_string())

# 3. Pass quality effect on zone 15 (back-right)
print("\n── Sim 3: Pass quality effect — Zone 15 ──")
pq_df = simulate_pass_quality_effect(markov_matrix, hitter_zone=15,
                                      set_loc=1, n=10_000)
print(pq_df.head(8).to_string())

# 4. Rally sequence simulation
print("\n── Sim 4: Rally sequence simulation (100 rallies) ──")
starting = [
    (1, 1, 11),  # good pass, outside, zone 11
    (1, 2, 15),  # good pass, oppo, zone 15
    (0, 1, 6),   # bad pass, outside, zone 6
]
rally_df = simulate_rally_sequence(markov_matrix, starting,
                                    n_rallies=100, max_touches=4)
print(rally_df.groupby("touch")["landing_zone"].describe().round(2).to_string())

# 5. Win probability by zone
print("\n── Sim 5: Win probability by hitter zone ──")
win_df = simulate_win_probability(train[train_mask])
print(win_df.sort_values("win_rate_%", ascending=False).to_string())
