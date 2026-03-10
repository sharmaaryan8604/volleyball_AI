import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# Zone bins: hitter_location & hit_land_location are continuous
# floats that represent court coordinates. Round to nearest integer
# to recover the zone ID (1-15 for hitter, 1-25 for landing).
# ------------------------------------------------------------------

HITTER_ZONE_RANGE  = (1, 15)   # 5-wide x 3-deep attack grid
LANDING_ZONE_RANGE = (1, 25)   # 5x5 full-court grid

# Encode categorical columns to integers so spatial_features
# and the Markov model can use them without crashing on int("in")
PASS_RATING_MAP = {"in": 1, "out": 0}
BLOCK_TOUCH_MAP = {"yes": 1, "no": 0}
SET_LOCATION_MAP = {
    "outside": 1,
    "oppo":    2,
    "quick":   3,
    "bic":     4,
    "dump":    5,
    "d-ball":  6,
    "in":      7,
    "blocked": 8,
}


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def clean_dataset(df):

    df = df.copy()

    # ----------------------------------------------------------
    # 1. Bin continuous location columns → integer zone IDs
    # ----------------------------------------------------------

    # hitter_location: continuous float → zone 1-15
    if "hitter_location" in df.columns:
        df["hitter_location"] = (
            pd.to_numeric(df["hitter_location"], errors="coerce")
            .round()
            .astype("Int64")
        )
        # keep only valid hitter zones
        df = df[
            df["hitter_location"].isna() |
            df["hitter_location"].between(*HITTER_ZONE_RANGE)
        ]

    # hit_land_location: continuous float → zone 1-25 (target)
    if "hit_land_location" in df.columns:
        df["hit_land_location"] = (
            pd.to_numeric(df["hit_land_location"], errors="coerce")
            .round()
            .astype("Int64")
        )
        df = df.dropna(subset=["hit_land_location"])
        df = df[df["hit_land_location"].between(*LANDING_ZONE_RANGE)]
        df["hit_land_location"] = df["hit_land_location"].astype(int)

    # Other numeric location columns
    for col in ["receive_location", "digger_location", "pass_land_location"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round()

    # ----------------------------------------------------------
    # 2. Encode categoricals → integers
    #    (keeps NaN as NaN so imputers downstream can handle it)
    # ----------------------------------------------------------

    if "pass_rating" in df.columns:
        df["pass_rating"] = df["pass_rating"].map(PASS_RATING_MAP)
        # NaN stays NaN for unmapped / missing values

    if "block_touch" in df.columns:
        df["block_touch"] = df["block_touch"].map(BLOCK_TOUCH_MAP)

    if "set_location" in df.columns:
        df["set_location"] = df["set_location"].map(SET_LOCATION_MAP)

    df = df.reset_index(drop=True)

    return df
