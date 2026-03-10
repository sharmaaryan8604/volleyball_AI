"""
simulation.py
-------------
Rally simulations and scenario analysis for the volleyball AI model.

Functions:
    simulate_landing_distribution  — Monte Carlo landing zone distribution
    simulate_scenario              — Compare outcomes under different conditions
    simulate_rally_sequence        — Chain multiple possessions into a rally
    simulate_pressure_effect       — How blockers affect landing distribution
    simulate_pass_quality_effect   — How pass quality shifts attack outcomes
    simulate_win_probability       — Estimate point-win % per hitter zone
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# 1. Basic landing zone distribution
# ─────────────────────────────────────────────────────────────

def simulate_landing_distribution(markov_matrix, hitter_zone,
                                   pass_rating=1, set_loc=1,
                                   n=10_000, random_state=42):
    """
    Monte Carlo: simulate n attacks from a given hitter zone.

    Args:
        markov_matrix : dict from build_transition_matrix
        hitter_zone   : int, zone 1-15
        pass_rating   : 0 (out) or 1 (in)
        set_loc       : int 1-8 (encoded set location)
        n             : number of simulated attacks
        random_state  : seed

    Returns:
        dict {landing_zone: count}, sorted by count descending
    """
    from src.markov_model import lookup_transition

    rng   = np.random.default_rng(random_state)
    probs = lookup_transition(markov_matrix, pass_rating, set_loc, hitter_zone)

    if not probs:
        raise ValueError(f"No Markov state found for hitter={hitter_zone}, "
                         f"pass={pass_rating}, set={set_loc}")

    zones = list(probs.keys())
    p     = np.array([probs[z] for z in zones], dtype=float)
    p    /= p.sum()

    samples = rng.choice(zones, size=n, p=p)
    counts  = defaultdict(int)
    for s in samples:
        counts[int(s)] += 1

    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────
# 2. Scenario comparison
# ─────────────────────────────────────────────────────────────

def simulate_scenario(markov_matrix, hitter_zone, scenarios, n=10_000):
    """
    Compare landing distributions across multiple scenarios.

    Args:
        markov_matrix : dict from build_transition_matrix
        hitter_zone   : int, attack zone
        scenarios     : list of dicts, each with keys:
                          'label'       : str description
                          'pass_rating' : 0 or 1
                          'set_loc'     : int 1-8
        n             : simulations per scenario

    Returns:
        pd.DataFrame  — rows=zones, cols=scenario labels (% share)

    Example:
        scenarios = [
            {'label': 'Good pass',  'pass_rating': 1, 'set_loc': 1},
            {'label': 'Bad pass',   'pass_rating': 0, 'set_loc': 1},
            {'label': 'Quick set',  'pass_rating': 1, 'set_loc': 3},
        ]
        df = simulate_scenario(P, hitter_zone=11, scenarios=scenarios)
    """
    results = {}
    for sc in scenarios:
        dist = simulate_landing_distribution(
            markov_matrix, hitter_zone,
            pass_rating=sc['pass_rating'],
            set_loc=sc['set_loc'],
            n=n
        )
        results[sc['label']] = dist

    all_zones = sorted(set(z for d in results.values() for z in d))
    df = pd.DataFrame(index=all_zones)

    for label, dist in results.items():
        df[label] = [dist.get(z, 0) / n * 100 for z in all_zones]

    df.index.name = "landing_zone"
    return df.round(2)


# ─────────────────────────────────────────────────────────────
# 3. Full rally sequence simulation
# ─────────────────────────────────────────────────────────────

def simulate_rally_sequence(markov_matrix, starting_states,
                             n_rallies=1000, max_touches=6,
                             random_state=42):
    """
    Simulate a sequence of possessions within a rally.
    Each possession: sample landing zone → becomes next hitter zone.

    Args:
        markov_matrix  : dict from build_transition_matrix
        starting_states: list of (pass_rating, set_loc, hitter_zone) tuples
                         — one is picked randomly each rally
        n_rallies      : number of rallies to simulate
        max_touches    : maximum possessions before stopping
        random_state   : seed

    Returns:
        pd.DataFrame with columns:
          rally_id, touch, hitter_zone, landing_zone, pass_rating, set_loc
    """
    from src.markov_model import lookup_transition

    rng  = np.random.default_rng(random_state)
    rows = []

    for rally_id in range(n_rallies):

        state = starting_states[rally_id % len(starting_states)]
        pass_r, set_l, hitter = state

        for touch in range(max_touches):

            probs = lookup_transition(markov_matrix, pass_r, set_l, hitter)
            if not probs:
                break

            zones = list(probs.keys())
            p     = np.array([probs[z] for z in zones], dtype=float)
            p    /= p.sum()

            landing = int(rng.choice(zones, p=p))

            rows.append({
                "rally_id":    rally_id,
                "touch":       touch + 1,
                "hitter_zone": hitter,
                "landing_zone":landing,
                "pass_rating": pass_r,
                "set_loc":     set_l,
            })

            # Next possession: landed zone becomes hitter zone
            hitter = landing
            # Vary pass quality randomly for realism
            pass_r = int(rng.choice([0, 1], p=[0.3, 0.7]))

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 4. Blocker pressure effect simulation
# ─────────────────────────────────────────────────────────────

def simulate_pressure_effect(ml_model, feature_template, hitter_zone,
                              blocker_counts=(0, 1, 2, 3), n=500):
    """
    Simulate how increasing blocker count shifts the predicted
    landing zone distribution using the ML model.

    Args:
        ml_model         : trained XGB or LGB pipeline
        feature_template : pd.Series — a real row from X_test to use as base
        hitter_zone      : int — fix hitter zone for fair comparison
        blocker_counts   : iterable of num_blockers values to test
        n                : copies per blocker count

    Returns:
        dict {n_blockers: {top_zone: probability}}
    """
    results = {}

    for nb in blocker_counts:
        rows = pd.DataFrame([feature_template] * n).reset_index(drop=True)
        rows["hitter_location"] = hitter_zone
        rows["num_blockers"]    = nb
        rows["attack_pressure"] = nb ** 2
        rows["quality_pressure"] = rows["pass_rating"] / (1 + nb)

        proba = ml_model.predict_proba(rows).mean(axis=0)
        top3  = np.argsort(proba)[::-1][:5]
        results[nb] = {int(z) + 1: round(float(proba[z]), 4) for z in top3}

    return results


# ─────────────────────────────────────────────────────────────
# 5. Pass quality effect simulation
# ─────────────────────────────────────────────────────────────

def simulate_pass_quality_effect(markov_matrix, hitter_zone,
                                  set_loc=1, n=5000):
    """
    Compare landing zone distributions between good pass (1) and bad pass (0).

    Returns:
        pd.DataFrame — zones × ['good_pass_%', 'bad_pass_%', 'shift_%']
    """
    good = simulate_landing_distribution(markov_matrix, hitter_zone,
                                          pass_rating=1, set_loc=set_loc, n=n)
    bad  = simulate_landing_distribution(markov_matrix, hitter_zone,
                                          pass_rating=0, set_loc=set_loc, n=n)

    all_zones = sorted(set(list(good.keys()) + list(bad.keys())))
    rows = []
    for z in all_zones:
        g = good.get(z, 0) / n * 100
        b = bad.get(z,  0) / n * 100
        rows.append({"zone": z, "good_pass_%": g,
                     "bad_pass_%": b, "shift_%": round(g - b, 2)})

    df = pd.DataFrame(rows).set_index("zone")
    return df.round(2).sort_values("good_pass_%", ascending=False)


# ─────────────────────────────────────────────────────────────
# 6. Win probability per zone
# ─────────────────────────────────────────────────────────────

def simulate_win_probability(train_df, zones=range(1, 16)):
    """
    Estimate the probability of winning a point given a hit from each zone.
    Uses historical win_reason and lose_reason data.

    Args:
        train_df : cleaned training DataFrame
        zones    : hitter zones to analyse

    Returns:
        pd.DataFrame — zone, win_rate, kill_rate, error_rate, n_attacks
    """
    rows = []

    for zone in zones:
        subset = train_df[train_df["hitter_location"] == zone]
        n = len(subset)
        if n == 0:
            continue

        win_mask   = subset["winning_team"] == subset["team"]
        kill_mask  = subset["win_reason"]   == "kill"
        error_mask = subset["lose_reason"].isin(["hit_error", "net", "error"])

        rows.append({
            "hitter_zone": zone,
            "n_attacks":   n,
            "win_rate_%":  round(win_mask.sum()   / n * 100, 1),
            "kill_rate_%": round(kill_mask.sum()  / n * 100, 1),
            "error_rate_%":round(error_mask.sum() / n * 100, 1),
        })

    return pd.DataFrame(rows).set_index("hitter_zone")
