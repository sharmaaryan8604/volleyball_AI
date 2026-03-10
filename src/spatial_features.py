import numpy as np
import pandas as pd


def add_spatial_features(df):
    """
    Compute spatial/tactical features from integer zone IDs.

    Requires clean_dataset() to have been called first so that:
      - hitter_location  is an integer in 1-15
      - set_location     is an integer in 1-8  (encoded from string)
      - pass_rating      is 0/1                (encoded from in/out)
      - hit_land_location is integer 1-25      (target, already clean)

    Court layout (zones 1-15, 5-wide x 3-deep):
      11 12 13 14 15   ← back row  (row 2)
       6  7  8  9 10   ← middle    (row 1)
       1  2  3  4  5   ← front row (row 0, near net)
    """

    df = df.copy()

    # ----------------------------------------------------------
    # Ensure numeric types (safe even if already numeric)
    # ----------------------------------------------------------

    for col in ["pass_rating", "num_blockers", "hitter_location", "set_location"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pass_rating"]  = df["pass_rating"].fillna(0)
    df["num_blockers"] = df["num_blockers"].fillna(0)

    # ----------------------------------------------------------
    # Hitter grid coordinates (col 0-4, row 0-2)
    # Zone layout: zone = row*5 + col + 1
    #   col = (zone-1) % 5    row = (zone-1) // 5
    # ----------------------------------------------------------

    df["hx"] = (df["hitter_location"] - 1) % 5    # 0=left … 4=right
    df["hy"] = (df["hitter_location"] - 1) // 5   # 0=front … 2=back

    # ----------------------------------------------------------
    # Setter grid coordinates (from encoded set_location 1-8)
    # set_location 1-8 maps to typical setter positions, not a
    # 5x5 grid — use a lookup instead of modular arithmetic
    # ----------------------------------------------------------

    SET_LOC_COORDS = {
        1: (4, 0),   # outside  → right front
        2: (4, 2),   # oppo     → right back
        3: (2, 0),   # quick    → middle front
        4: (3, 0),   # bic      → right-centre front
        5: (2, 0),   # dump     → middle front (setter dumps)
        6: (2, 1),   # d-ball   → middle centre
        7: (2, 0),   # in       → middle front
        8: (2, 0),   # blocked  → middle front
    }

    def _set_coord(val, axis):
        if pd.isna(val):
            return np.nan
        return SET_LOC_COORDS.get(int(val), (2, 0))[axis]

    df["sx"] = df["set_location"].apply(lambda v: _set_coord(v, 0))
    df["sy"] = df["set_location"].apply(lambda v: _set_coord(v, 1))

    # ----------------------------------------------------------
    # Distance features
    # ----------------------------------------------------------

    df["dist_net"]    = df["hy"]                           # rows from net
    df["dist_center"] = np.sqrt(
        (df["hx"] - 2) ** 2 +
        (df["hy"] - 1) ** 2                               # centre of 3-row grid
    )

    # ----------------------------------------------------------
    # Attack vector (setter → hitter)
    # ----------------------------------------------------------

    df["attack_dx"]       = df["hx"] - df["sx"]
    df["attack_dy"]       = df["hy"] - df["sy"]
    df["attack_distance"] = np.sqrt(df["attack_dx"]**2 + df["attack_dy"]**2)
    df["attack_angle"]    = np.arctan2(df["attack_dy"], df["attack_dx"] + 1e-5)

    # ----------------------------------------------------------
    # Attack trajectory (expected direction from hitter)
    # ----------------------------------------------------------

    df["traj_dx"]       = df["hx"] - 2          # deviation from centre column
    df["traj_dy"]       = df["hy"] - 2           # deviation from back row
    df["traj_distance"] = np.sqrt(df["traj_dx"]**2 + df["traj_dy"]**2)
    df["traj_angle"]    = np.arctan2(df["traj_dy"], df["traj_dx"] + 1e-5)

    # ----------------------------------------------------------
    # Tactical indicators
    # ----------------------------------------------------------

    df["deep_attack"]  = (df["hy"].fillna(-1) >= 2).astype(int)
    df["short_attack"] = (df["hy"].fillna(-1) == 0).astype(int)
    df["cross_court"]  = (df["hx"].fillna(-1) >= 3).astype(int)
    df["line_attack"]  = (df["hx"].fillna(-1) <= 1).astype(int)
    df["back_row"]     = (df["hy"].fillna(-1) >= 2).astype(int)

    # ----------------------------------------------------------
    # Pressure features
    # ----------------------------------------------------------

    df["attack_pressure"] = df["num_blockers"] ** 2
    df["quality_pressure"] = df["pass_rating"] / (1 + df["num_blockers"])

    return df
