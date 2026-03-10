import numpy as np
import pandas as pd


def add_spatial_features(df):

    numeric_cols = [
        "pass_rating",
        "num_blockers",
        "hitter_location",
        "set_location"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pass_rating"] = df["pass_rating"].fillna(0)
    df["num_blockers"] = df["num_blockers"].fillna(0)

    # --------------------------------
    # Hitter coordinates (court grid)
    # --------------------------------

    df["hx"] = (df["hitter_location"] - 1) % 5
    df["hy"] = (df["hitter_location"] - 1) // 5

    # --------------------------------
    # Setter coordinates
    # --------------------------------

    df["sx"] = (df["set_location"] - 1) % 5
    df["sy"] = (df["set_location"] - 1) // 5

    # --------------------------------
    # Distance features
    # --------------------------------

    df["dist_net"] = df["hy"]

    df["dist_center"] = np.sqrt(
        (df["hx"] - 2) ** 2 +
        (df["hy"] - 2) ** 2
    )

    # --------------------------------
    # Attack vector (setter → hitter)
    # --------------------------------

    df["attack_dx"] = df["hx"] - df["sx"]
    df["attack_dy"] = df["hy"] - df["sy"]

    df["attack_distance"] = np.sqrt(
        df["attack_dx"] ** 2 +
        df["attack_dy"] ** 2
    )

    df["attack_angle"] = np.arctan2(
        df["attack_dy"],
        df["attack_dx"] + 1e-5
    )

    # --------------------------------
# Attack trajectory modelling
# --------------------------------

# Expected landing direction from hitter

    df["traj_dx"] = df["hx"] - 2
    df["traj_dy"] = df["hy"] - 4

# trajectory distance
    df["traj_distance"] = np.sqrt(
    df["traj_dx"]**2 +
    df["traj_dy"]**2
)

# trajectory angle
    df["traj_angle"] = np.arctan2(
    df["traj_dy"],
    df["traj_dx"] + 1e-5
)

# deep attack indicator
    df["deep_attack"] = (df["hy"] >= 3).astype(int)

# short attack indicator
    df["short_attack"] = (df["hy"] <= 1).astype(int)
    # --------------------------------
    # Tactical attack indicators
    # --------------------------------

    df["cross_court"] = (df["hx"] > 2).astype(int)
    df["line_attack"] = (df["hx"] <= 2).astype(int)
    df["back_row"] = (df["hy"] >= 3).astype(int)

    # --------------------------------
    # Pressure features
    # --------------------------------

    df["attack_pressure"] = df["num_blockers"] ** 2

    df["quality_pressure"] = (
        df["pass_rating"] /
        (1 + df["num_blockers"])
    )

    return df