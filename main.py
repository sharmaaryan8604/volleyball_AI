import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, clean_dataset
from src.spatial_features import add_spatial_features
from src.ml_model import train_ml_model
from src.markov_model import build_transition_matrix


# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

print("Loading data...")

train, test = load_data(
    "data/training data.csv",
    "data/testing data.csv"
    
)

train = clean_dataset(train)
test = clean_dataset(test)

train = add_spatial_features(train)
test = add_spatial_features(test)


# -----------------------------------------------------
# Feature Selection
# -----------------------------------------------------

features = [
    "receive_location",
    "digger_location",
    "pass_land_location",
    "hitter_location",
    "set_location",
    "pass_rating",
    "set_type",
    "hit_type",
    "serve_type",
    "num_blockers",
    "block_touch",
    "rally",
    "round",

    # spatial features
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

# XGBoost requires labels starting at 0
y = train["hit_land_location"] - 1


# -----------------------------------------------------
# Train/Test Split
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------------------
# Train ML Models
# -----------------------------------------------------

print("\nTraining ML models...")

xgb_model, lgb_model = train_ml_model(X_train, y_train)

print("Models trained.")


# -----------------------------------------------------
# Build Markov Sequence Model
# -----------------------------------------------------

print("\nBuilding Markov model...")

markov_matrix = build_transition_matrix(train)


# -----------------------------------------------------
# Zone Priors
# -----------------------------------------------------

zone_counts = train["hit_land_location"].value_counts(normalize=True)

zone_prior = {int(k) - 1: float(v) for k, v in zone_counts.items()}


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

print("\nEvaluating model...")

xgb_probs = xgb_model.predict_proba(X_test)
lgb_probs = lgb_model.predict_proba(X_test)

# Ensemble prediction
probs = 0.5 * xgb_probs + 0.5 * lgb_probs

# # -----------------------------------
# # Bayesian probability smoothing
# # -----------------------------------

# alpha = 0.05  # smoothing strength

# zone_freq = train["hit_land_location"].value_counts(normalize=True)

# zone_freq = {int(k)-1: v for k, v in zone_freq.items()}

# for i in range(probs.shape[0]):
#     for z in range(probs.shape[1]):

#         prior = zone_freq.get(z, 0)

#         probs[i, z] = (
#             (1 - alpha) * probs[i, z] +
#             alpha * prior
#         )
# -----------------------------------------------------
# Hybrid Prediction (ML + Markov + Priors)
# -----------------------------------------------------

for i, row in X_test.iterrows():

    idx = X_test.index.get_loc(i)

    pass_rating = row.get("pass_rating")
    set_loc = row.get("set_location")
    hitter = row.get("hitter_location")

    if pd.isna(pass_rating) or pd.isna(set_loc) or pd.isna(hitter):
        continue

    state = (
        int(pass_rating),
        int(set_loc),
        int(hitter)
    )

    # Markov blending
    if state in markov_matrix:

        for zone, p in markov_matrix[state].items():

            zone_idx = int(zone) - 1

            if zone_idx < probs.shape[1]:

                probs[idx, zone_idx] = (
                    0.70 * probs[idx, zone_idx] +
                    0.30 * p
                )

    # Zone prior blending
    prior_weight = 0.10

    for zone, p in zone_prior.items():

        if zone < probs.shape[1]:

            probs[idx, zone] = (
                (1 - prior_weight) * probs[idx, zone]
                + prior_weight * p
            )


# -----------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------

top1 = top_k_accuracy_score(y_test, probs, k=1)
top3 = top_k_accuracy_score(y_test, probs, k=3)
top5 = top_k_accuracy_score(y_test, probs, k=5)


# -----------------------------------------------------
# Safe MRR
# -----------------------------------------------------

ranks = []

for i in range(len(y_test)):

    sorted_idx = np.argsort(probs[i])[::-1]

    true_zone = int(y_test.iloc[i])

    match = np.where(sorted_idx == true_zone)[0]

    if len(match) > 0:
        rank = match[0] + 1
        ranks.append(1 / rank)
    else:
        ranks.append(0)

mrr = np.mean(ranks)


# -----------------------------------------------------
# Results
# -----------------------------------------------------

print("\n===== FINAL RESULTS =====\n")

print("Top1 Accuracy:", round(top1 * 100, 2))
print("Top3 Accuracy:", round(top3 * 100, 2))
print("Top5 Accuracy:", round(top5 * 100, 2))
print("MRR:", round(mrr, 4))



from src.markov_model import build_transition_matrix
from src.attack_map import plot_attack_map

# Build Markov matrix
P = build_transition_matrix(train)

# Plot attack directions
plot_attack_map(P)
