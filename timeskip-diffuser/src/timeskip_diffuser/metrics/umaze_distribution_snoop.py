import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Define the two corner regions of the U
# (Replace with your exact coordinates)
# ---------------------------------------------------------

BOTTOM_LEFT_X_RANGE  = (-1.5, -0.5)
BOTTOM_LEFT_Y_RANGE  = (-1.5, -0.5)

TOP_RIGHT_X_RANGE    = (-1.5, -0.5)
TOP_RIGHT_Y_RANGE    = (0.5,   1.5)


def in_region(x, y, xr, yr):
    return (xr[0] <= x <= xr[1]) and (yr[0] <= y <= yr[1])


def is_bottom_left(x, y):
    return in_region(x, y, BOTTOM_LEFT_X_RANGE, BOTTOM_LEFT_Y_RANGE)


def is_top_right(x, y):
    return in_region(x, y, TOP_RIGHT_X_RANGE, TOP_RIGHT_Y_RANGE)
def compute_u_completion_rate(dataset, allow_reverse=True):
    """
    dataset: UMazeFlatDataset
    allow_reverse: if True, also counts top-right -> bottom-left
    """

    u_count = 0
    total = len(dataset.trajectories)

    for traj in dataset.trajectories:
        start_x, start_y = traj[0]
        end_x, end_y = traj[-1]

        forward = is_bottom_left(start_x, start_y) and is_top_right(end_x, end_y)
        reverse = is_top_right(start_x, start_y) and is_bottom_left(end_x, end_y)

        if forward or (allow_reverse and reverse):
            u_count += 1

    rate = u_count / total
    direction = "both directions" if allow_reverse else "forward only"

    print(f"U-completion rate ({direction}): {rate:.4f}  ({u_count}/{total})")
    return rate, u_count, total

def visualize_u_trajectories(dataset, allow_reverse=True):
    """
    Plots all start and end points of trajectories
    and highlights which ones satisfy U-completion.
    """

    starts_x = []
    starts_y = []
    ends_x = []
    ends_y = []

    u_starts_x = []
    u_starts_y = []
    u_ends_x = []
    u_ends_y = []

    for traj in dataset.trajectories:
        sx, sy = traj[0]
        ex, ey = traj[-1]

        starts_x.append(sx)
        starts_y.append(sy)
        ends_x.append(ex)
        ends_y.append(ey)

        forward = is_bottom_left(sx, sy) and is_top_right(ex, ey)
        reverse = is_top_right(sx, sy) and is_bottom_left(ex, ey)

        if forward or (allow_reverse and reverse):
            u_starts_x.append(sx)
            u_starts_y.append(sy)
            u_ends_x.append(ex)
            u_ends_y.append(ey)

    plt.figure(figsize=(6,6))
    plt.scatter(starts_x, starts_y, s=10, c="blue", alpha=0.5, label="All starts")
    plt.scatter(ends_x, ends_y, s=10, c="red", alpha=0.5, label="All ends")

    # Highlight U-complete start/end points
    if len(u_starts_x) > 0:
        plt.scatter(u_starts_x, u_starts_y, s=60, c="lime", edgecolor="black", label="U-starts")
        plt.scatter(u_ends_x, u_ends_y, s=60, c="gold", edgecolor="black", label="U-ends")

    # Plot the BL and TR regions as rectangles
    bl = plt.Rectangle(
        (BOTTOM_LEFT_X_RANGE[0], BOTTOM_LEFT_Y_RANGE[0]),
        BOTTOM_LEFT_X_RANGE[1]-BOTTOM_LEFT_X_RANGE[0],
        BOTTOM_LEFT_Y_RANGE[1]-BOTTOM_LEFT_Y_RANGE[0],
        edgecolor="cyan", facecolor="cyan", alpha=0.2, label="Bottom-left region"
    )
    plt.gca().add_patch(bl)

    tr = plt.Rectangle(
        (TOP_RIGHT_X_RANGE[0], TOP_RIGHT_Y_RANGE[0]),
        TOP_RIGHT_X_RANGE[1]-TOP_RIGHT_X_RANGE[0],
        TOP_RIGHT_Y_RANGE[1]-TOP_RIGHT_Y_RANGE[0],
        edgecolor="magenta", facecolor="magenta", alpha=0.2, label="Top-right region"
    )
    plt.gca().add_patch(tr)

    plt.title("U-Completion Start/End Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
