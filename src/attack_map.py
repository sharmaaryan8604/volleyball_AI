import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# Court layout — matches DataVolley standard zone map
#
#  Court is HORIZONTAL. Net runs VERTICALLY down the centre.
#
#  Left half  = attack side  (own team)   zones 1-15
#  Right half = landing side (opponent)   zones 1-15
#
#  Outer blue boundary zones 16-26 surround the inner green area.
#
#  Inner green grid — 3 cols × 5 rows per half:
#
#    Attack side (left):        Landing side (right):
#    col→  0   1   2            col→  3   4   5
#    row 0 (top):  5  10  15        11   6   1
#    row 1:        4   9  14        12   7   2
#    row 2:        3   8  13        13   8   3
#    row 3:        2   7  12        14   9   4
#    row 4 (bot):  1   6  11        15  10   5
#
#  Each inner cell: 3m wide × 3m tall
#  Outer boundary:  2m wide (sides) / 2m tall (top/bottom)
# ─────────────────────────────────────────────────────────────

# Dimensions (in plot units = metres)
INNER_CW = 3.0   # inner cell width
INNER_CH = 3.0   # inner cell height
OUTER_W  = 2.0   # outer boundary width
NET_W    = 0.3   # net thickness visual

INNER_COLS = 3   # cols per half
INNER_ROWS = 5

# Total court
COURT_W = 2 * OUTER_W + 2 * INNER_COLS * INNER_CW + NET_W
COURT_H = 2 * OUTER_W + INNER_ROWS * INNER_CH

# X offsets
X_LEFT_INNER  = OUTER_W                                    # left inner starts here
X_NET         = OUTER_W + INNER_COLS * INNER_CW            # net centre
X_RIGHT_INNER = X_NET + NET_W                              # right inner starts here

# Y offset
Y_INNER = OUTER_W                                          # inner bottom


def _cell(col, row):
    """Bottom-left corner of an inner cell (col 0-5, row 0-4 from bottom)."""
    if col < INNER_COLS:
        x = X_LEFT_INNER + col * INNER_CW
    else:
        x = X_RIGHT_INNER + (col - INNER_COLS) * INNER_CW
    y = Y_INNER + row * INNER_CH
    return x, y


def _cell_centre(col, row):
    x, y = _cell(col, row)
    return x + INNER_CW / 2, y + INNER_CH / 2


# Zone → (col, row) for attack side — row 0 = bottom, row 4 = top
ATTACK_GRID = {
     1: (0, 0),  2: (0, 1),  3: (0, 2),  4: (0, 3),  5: (0, 4),
     6: (1, 0),  7: (1, 1),  8: (1, 2),  9: (1, 3), 10: (1, 4),
    11: (2, 0), 12: (2, 1), 13: (2, 2), 14: (2, 3), 15: (2, 4),
}

# Zone → (col, row) for landing side (mirrored — zone 1 is nearest net)
LANDING_GRID = {
    11: (3, 4), 12: (3, 3), 13: (3, 2), 14: (3, 1), 15: (3, 0),
     6: (4, 4),  7: (4, 3),  8: (4, 2),  9: (4, 1), 10: (4, 0),
     1: (5, 4),  2: (5, 3),  3: (5, 2),  4: (5, 1),  5: (5, 0),
}

# Outer boundary zones with their centre positions
def _outer_centres():
    centres = {}
    # Top row (above inner, left to right): 23,24,25 | 26 | 16,17,18,19
    top_y = Y_INNER + INNER_ROWS * INNER_CH + OUTER_W / 2
    top_zones_left  = [23, 24, 25]
    top_zones_right = [16, 17, 18, 19]
    for i, z in enumerate(top_zones_left):
        centres[z] = (X_LEFT_INNER + i * INNER_CW + INNER_CW/2, top_y)
    centres[26] = (X_NET + NET_W/2, top_y)
    for i, z in enumerate(top_zones_right):
        centres[z] = (X_RIGHT_INNER + i * INNER_CW + INNER_CW/2, top_y)

    # Bottom row: 19,18,17,16 | 26 | 25,24,23
    bot_y = OUTER_W / 2
    bot_zones_left  = [19, 18, 17, 16]
    bot_zones_right = [26, 25, 24, 23]
    for i, z in enumerate(bot_zones_left):
        centres[z] = (X_LEFT_INNER + i * INNER_CW + INNER_CW/2 - INNER_CW, bot_y)
    for i, z in enumerate(bot_zones_right):
        centres[z] = (X_RIGHT_INNER + i * INNER_CW + INNER_CW/2 - INNER_CW, bot_y)

    # Left col: 22,21,20,19 (top to bottom)
    left_x = OUTER_W / 2
    for i, z in enumerate([22, 21, 20, 19]):
        centres[z] = (left_x,
                      Y_INNER + (INNER_ROWS - 1 - i) * INNER_CH + INNER_CH/2)

    # Right col: 20,21,22,23
    right_x = X_RIGHT_INNER + INNER_COLS * INNER_CW + OUTER_W / 2
    for i, z in enumerate([20, 21, 22, 23]):
        centres[z] = (right_x,
                      Y_INNER + (INNER_ROWS - 1 - i) * INNER_CH + INNER_CH/2)
    return centres


ATTACK_ORIGINS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Perceptually distinct colors — chosen so no two adjacent zones
# share a similar hue, and all are visible on dark green background
ZONE_COLORS = {
     6: "#FF4136",   # vivid red
     7: "#FF851B",   # orange
     8: "#FFDC00",   # yellow
     9: "#01FF70",   # neon green
    10: "#7FDBFF",   # sky blue
    11: "#0074D9",   # strong blue
    12: "#B10DC9",   # purple
    13: "#F012BE",   # magenta
    14: "#FFFFFF",   # white
    15: "#00FFFF",   # cyan
}


def _draw_court(ax):
    # Outer blue background (full court)
    ax.add_patch(plt.Rectangle((0, 0), COURT_W, COURT_H,
                 lw=2, edgecolor="#1a5276", facecolor="#2e86c1", zorder=0))

    # Inner green — left half
    ax.add_patch(plt.Rectangle(
        (X_LEFT_INNER, Y_INNER),
        INNER_COLS * INNER_CW, INNER_ROWS * INNER_CH,
        lw=0, facecolor="#1e8449", zorder=1))

    # Inner green — right half
    ax.add_patch(plt.Rectangle(
        (X_RIGHT_INNER, Y_INNER),
        INNER_COLS * INNER_CW, INNER_ROWS * INNER_CH,
        lw=0, facecolor="#1e8449", zorder=1))

    # Inner grid lines — left half
    for col in range(INNER_COLS + 1):
        x = X_LEFT_INNER + col * INNER_CW
        ax.plot([x, x], [Y_INNER, Y_INNER + INNER_ROWS*INNER_CH],
                color="white", lw=0.8, zorder=2)
    for row in range(INNER_ROWS + 1):
        y = Y_INNER + row * INNER_CH
        ax.plot([X_LEFT_INNER, X_LEFT_INNER + INNER_COLS*INNER_CW], [y, y],
                color="white", lw=0.8, zorder=2)

    # Inner grid lines — right half
    for col in range(INNER_COLS + 1):
        x = X_RIGHT_INNER + col * INNER_CW
        ax.plot([x, x], [Y_INNER, Y_INNER + INNER_ROWS*INNER_CH],
                color="white", lw=0.8, zorder=2)
    for row in range(INNER_ROWS + 1):
        y = Y_INNER + row * INNER_CH
        ax.plot([X_RIGHT_INNER, X_RIGHT_INNER + INNER_COLS*INNER_CW], [y, y],
                color="white", lw=0.8, zorder=2)

    # Net (vertical red dashed line)
    net_x = X_NET + NET_W / 2
    ax.plot([net_x, net_x], [Y_INNER, Y_INNER + INNER_ROWS*INNER_CH],
            color="#e74c3c", lw=3, linestyle="--", zorder=5)

    # Outer border lines
    ax.add_patch(plt.Rectangle((0, 0), COURT_W, COURT_H,
                 lw=2.5, edgecolor="white", facecolor="none", zorder=3))

    # Zone labels — attack side
    for zone, (col, row) in ATTACK_GRID.items():
        cx, cy = _cell_centre(col, row)
        ax.text(cx, cy, str(zone), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)

    # Zone labels — landing side
    for zone, (col, row) in LANDING_GRID.items():
        cx, cy = _cell_centre(col, row)
        ax.text(cx, cy, str(zone), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)



    # Side labels
    ax.text(X_LEFT_INNER + INNER_COLS*INNER_CW/2,
            Y_INNER + INNER_ROWS*INNER_CH + OUTER_W*0.6,
            "ATTACK SIDE", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)
    ax.text(X_RIGHT_INNER + INNER_COLS*INNER_CW/2,
            Y_INNER + INNER_ROWS*INNER_CH + OUTER_W*0.6,
            "LANDING SIDE", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)

    ax.set_xlim(0, COURT_W)
    ax.set_ylim(0, COURT_H)
    ax.set_aspect("equal")
    ax.axis("off")


def _flatten_markov(P) -> dict:
    import pandas as pd
    from collections import defaultdict

    if isinstance(P, pd.DataFrame):
        return {start: dict(P.loc[start]) for start in P.index}

    accum = defaultdict(lambda: defaultdict(list))
    for state, zone_probs in P.items():
        hitter = state[-1]
        if hitter is None:
            continue
        for zone, prob in zone_probs.items():
            accum[hitter][zone].append(prob)

    return {
        hitter: {zone: float(np.mean(probs)) for zone, probs in zd.items()}
        for hitter, zd in accum.items()
    }


def plot_attack_map(P, threshold=0.06):
    """
    Plot attack trajectories on a DataVolley-style horizontal court.
    Net runs vertically in the centre.
    Arrows go from attack-side zones (left) to landing zones (right).
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#1a2530")
    ax.set_facecolor("#1a2530")

    _draw_court(ax)

    flat           = _flatten_markov(P)
    legend_handles = []
    total_arrows   = 0

    for start_zone in ATTACK_ORIGINS:
        if start_zone not in flat:
            continue
        if start_zone not in ATTACK_GRID:
            continue

        color          = ZONE_COLORS.get(start_zone, "#e74c3c")
        col_s, row_s   = ATTACK_GRID[start_zone]
        sx, sy         = _cell_centre(col_s, row_s)
        drawn          = 0

        for end_zone, prob in sorted(flat[start_zone].items(),
                                     key=lambda kv: kv[1], reverse=True):
            if prob < threshold:
                continue
            if end_zone not in LANDING_GRID:
                continue

            col_e, row_e = LANDING_GRID[end_zone]
            ex, ey       = _cell_centre(col_e, row_e)

            ax.annotate(
                "",
                xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=min(prob * 4 + 0.25, 0.92),
                    lw=max(prob * 12, 1.2),
                    mutation_scale=14,
                ),
                zorder=6
            )
            drawn        += 1
            total_arrows += 1

        if drawn > 0:
            legend_handles.append(
                mpatches.Patch(color=color,
                               label=f"Zone {start_zone}  ({drawn} targets)")
            )

    ax.set_title(
        f"Volleyball Attack Trajectories  │  "
        f"Origins: {', '.join(map(str, ATTACK_ORIGINS))}  │  "
        f"threshold ≥ {int(threshold*100)}%  │  "
        f"{total_arrows} arrows",
        fontsize=11, fontweight="bold", pad=12, color="white"
    )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=len(legend_handles),
            fontsize=9, framealpha=0.85,
            title="Attack origin zones", title_fontsize=9
        )

    plt.tight_layout()
    plt.show()