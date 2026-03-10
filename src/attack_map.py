import matplotlib.pyplot as plt


# ----------------------------
# Bottom court zones (attack origins)
# ----------------------------

BOTTOM_ZONES = {
11:(0,4),6:(1,4),1:(2,4),
12:(0,3),7:(1,3),2:(2,3),
13:(0,2),8:(1,2),3:(2,2),
14:(0,1),9:(1,1),4:(2,1),
15:(0,0),10:(1,0),5:(2,0)
}


# ----------------------------
# Top court zones (landing)
# ----------------------------

TOP_ZONES = {
11:(0,10),6:(1,10),1:(2,10),
12:(0,9),7:(1,9),2:(2,9),
13:(0,8),8:(1,8),3:(2,8),
14:(0,7),9:(1,7),4:(2,7),
15:(0,6),10:(1,6),5:(2,6)
}


# ----------------------------
# Allowed attack zones
# ----------------------------

ATTACK_ORIGINS = [1,2,6,7,11,12]


# ----------------------------
# Draw court grid
# ----------------------------

def draw_court(ax):

    for x in range(4):
        ax.plot([x,x],[0,11],'black')

    for y in range(12):
        ax.plot([0,3],[y,y],'black')

    # net
    ax.plot([0,3],[5,5],linewidth=3)


# ----------------------------
# Plot attack trajectories
# ----------------------------

def plot_attack_map(P, threshold=0.05):

    fig, ax = plt.subplots(figsize=(6,12))

    draw_court(ax)

    # bottom zones
    for z,(x,y) in BOTTOM_ZONES.items():

        ax.text(x+0.5,y+0.5,str(z),
                ha="center",va="center")

    # top zones
    for z,(x,y) in TOP_ZONES.items():

        ax.text(x+0.5,y+0.5,str(z),
                ha="center",va="center")

    # attack trajectories
    for start_zone in P.index:

        # only allow attacks from selected zones
        if start_zone not in ATTACK_ORIGINS:
            continue

        for end_zone in P.columns:

            prob = P.loc[start_zone,end_zone]

            if prob < threshold:
                continue

            if end_zone not in TOP_ZONES:
                continue

            x1,y1 = BOTTOM_ZONES[start_zone]
            x2,y2 = TOP_ZONES[end_zone]

            ax.arrow(
                x1+0.5,
                y1+0.5,
                (x2-x1),
                (y2-y1),
                head_width=0.15,
                alpha=prob,
                color="red"
            )

    ax.set_xlim(0,3)
    ax.set_ylim(0,11)

    ax.set_title("Volleyball Attack Trajectories (Zones 1,2,6,7,11,12)")

    ax.axis("off")

    plt.show()