from complex import Complex
from typing import List, Tuple, Iterable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np


def animate_complex(
    sequence: Iterable[Complex | complex | tuple[float, float]],
    *,
    interval: int = 200,
) -> animation.FuncAnimation:
    """
    Animate a sequence of complex‑number samples in the 2‑D plane.

    Parameters
    ----------
    sequence : iterable of `Complex`, Python `complex`, or (x,y) tuples
    interval : delay between frames in **ms**

    Returns
    -------
    matplotlib.animation.FuncAnimation – handy if you need to save().
    """

    # Convert the iterable to a concrete list (needed to pre‑fit axes)
    seq: List[Complex] = [z if isinstance(z, Complex) else Complex(z) for z in sequence]

    # Pre‑compute limits for a clean box that fits everything
    xs = [z.re for z in seq]
    ys = [z.im for z in seq]
    span = max(max(map(abs, xs)), max(map(abs, ys)), 1.0)
    margin = 0.1 * span

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-span - margin, span + margin)
    ax.set_ylim(-span - margin, span + margin)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Complex number animation")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Artists
    point, = ax.plot([], [], "ro", markersize=6)
    trail, = ax.plot([], [], "b-", alpha=0.5, linewidth=1)

    # History containers for the trail
    history_x: List[float] = []
    history_y: List[float] = []

    def init():
        point.set_data([], [])
        trail.set_data([], [])
        return point, trail

    def update(frame: int):
        z = seq[frame]
        history_x.append(z.re)
        history_y.append(z.im)

        point.set_data(z.re, z.im)
        trail.set_data(history_x, history_y)
        ax.set_title(f"t = {frame}  |  z = {z.re:+.3f} {z.im:+.3f}i")
        return point, trail

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(seq),
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )
    plt.show()
    return anim


if __name__ == "__main__":
    o = [Complex(0)]
    for i in range(1, 360):
        o.append(o[-1]*Complex(math.pi/180))
    animate_complex(o, interval=1)