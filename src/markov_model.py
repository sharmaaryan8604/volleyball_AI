import pandas as pd
from collections import defaultdict


def build_transition_matrix(df, min_samples=5):
    """
    Build a Markov transition matrix:
        (pass_rating, set_location, hitter_location) -> hit_land_location

    Requires clean_dataset() to have run first so that:
      - pass_rating     is 0/1   (was "in"/"out")
      - set_location    is 1-8   (was "outside"/"quick" etc.)
      - hitter_location is 1-15  (rounded integer zone)
      - hit_land_location is 1-25 (rounded integer zone)

    Falls back gracefully when a state has too few samples.

    Returns:
        dict: {state_tuple -> {landing_zone: probability}}
    """

    required = ["pass_rating", "set_location", "hitter_location", "hit_land_location"]
    data = df[required].copy()

    # Convert all to numeric — safe because preprocessing already encoded strings
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()

    # Round to integers to ensure consistent grouping keys
    for col in required:
        data[col] = data[col].round().astype(int)

    def _compute_probs(group_cols):
        matrix = defaultdict(dict)
        for state, group in data.groupby(group_cols)["hit_land_location"]:
            if len(group) < min_samples:
                continue
            counts = group.value_counts()
            total  = counts.sum()
            state_key = state if isinstance(state, tuple) else (state,)
            for zone, count in counts.items():
                matrix[state_key][int(zone)] = count / total
        return matrix

    # Three levels of conditioning
    primary   = _compute_probs(["pass_rating", "set_location", "hitter_location"])
    fallback1 = _compute_probs(["set_location", "hitter_location"])
    fallback2 = _compute_probs(["hitter_location"])

    matrix = {}

    for state, probs in primary.items():
        matrix[state] = probs                          # (pass, set, hitter)

    for state, probs in fallback1.items():
        key = (None,) + state                          # (None, set, hitter)
        if key not in matrix:
            matrix[key] = probs

    for state, probs in fallback2.items():
        key = (None, None) + state                     # (None, None, hitter)
        if key not in matrix:
            matrix[key] = probs

    return matrix


def lookup_transition(matrix, pass_rating, set_loc, hitter):
    """
    Look up transition probabilities with graceful fallback chain.

    Args:
        matrix      : dict returned by build_transition_matrix
        pass_rating : encoded int (0 or 1)
        set_loc     : encoded int (1-8)
        hitter      : integer zone (1-15)

    Returns:
        dict {landing_zone: probability} or {} if no match
    """

    def _safe_int(v):
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return None

    pr = _safe_int(pass_rating)
    sl = _safe_int(set_loc)
    ht = _safe_int(hitter)

    if ht is None:
        return {}

    # Full state
    if pr is not None and sl is not None:
        key = (pr, sl, ht)
        if key in matrix:
            return matrix[key]

    # Drop pass_rating
    if sl is not None:
        key = (None, sl, ht)
        if key in matrix:
            return matrix[key]

    # Hitter only
    key = (None, None, ht)
    return matrix.get(key, {})
