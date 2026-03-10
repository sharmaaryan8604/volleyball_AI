import pandas as pd


def build_transition_matrix(df):
    """
    Build Markov transition probability matrix
    from hitter_location -> hit_land_location
    """

    # create transition counts
    transitions = pd.crosstab(
        df["hitter_location"],
        df["hit_land_location"]
    )

    # convert to probabilities
    P = transitions.div(transitions.sum(axis=1), axis=0)

    return P