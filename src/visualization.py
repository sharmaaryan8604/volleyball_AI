


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ZONE_COORDS = {
11:(0,4), 6:(1,4), 1:(2,4),
12:(0,3), 7:(1,3), 2:(2,3),
13:(0,2), 8:(1,2), 3:(2,2),
14:(0,1), 9:(1,1), 4:(2,1),
15:(0,0),10:(1,0),5:(2,0)
}

def plot_heatmap(zone_probs):

    grid = np.zeros((5,3))

    for zone,p in zone_probs.items():

        if zone not in ZONE_COORDS:
            continue

        x,y = ZONE_COORDS[zone]

        grid[4-y,x] = p

    sns.heatmap(grid,annot=True,cmap="YlOrRd")

    plt.title("Volleyball Attack Landing Zones")
    plt.show()