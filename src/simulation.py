import numpy as np

def simulate_rally(P,start_zone,n=5000):

    probs = P.loc[start_zone].values

    probs = probs / probs.sum()

    zones = P.columns

    sim = np.random.choice(zones,size=n,p=probs)

    return sim