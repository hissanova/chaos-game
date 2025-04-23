from typing import Callable, Dict, List, NamedTuple, Sequence
from pathlib import Path
from random import randint
from math import acos, cos, pi, sin, sqrt


import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns
# import plotly.graph_objects as go


def get_polygon(n):
    return [(float(sin(2 * i * pi / n)), float(cos(2 * i * pi / n)))
            for i in range(n)]


def proceed_euc(fixed, point, r):
    return (1 - r) * fixed + r * point


def proceed_sphere(fixed, point, r):
    # a,b:vectors, r:dividing ratio
    vertical = np.cross(np.cross(fixed, point), fixed)
    vertical /= np.linalg.norm(vertical)
    t1 = r * get_angle(fixed, point)
    return cos(t1) * fixed + sin(t1) * vertical


# --- stereographic projection --- #
def proj_plane(x, z):
    return x / (1 + z)


def stereoG3D(v):
    return [proj_plane(v[0], v[2]), proj_plane(v[1], v[2])]


def get_angle(a, b):
    return float(acos(np.dot(a, b)))


class Game(NamedTuple):
    fixed_points: Sequence[np.array]
    ratio: float
    rounds: int
    points: Sequence[np.array]


IndexSelector = Callable[[Sequence[np.array]], int]


def get_random_selector(n: int) -> IndexSelector:

    def _selector(prev_points: Sequence[np.array]) -> int:
        return randint(0, n - 1)

    return _selector


def create_game(
    start: np.array,
    fixed_points: Sequence[np.array],
    proceed_func: Callable,
    selector: IndexSelector,
    ratio: float,
    rounds: int,
):
    points = [start]
    for i in tqdm(range(rounds)):
        index = selector(points)
        next_point = proceed_func(fixed_points[index], points[i], ratio)
        next_point = np.array([next_point])
        points = np.append(points, next_point, axis=0)
    return Game(
        fixed_points=fixed_points,
        ratio=ratio,
        rounds=rounds,
        points=points,
    )


class PointConfig(NamedTuple):
    size: int
    color: str


def _plot_points(ax, points, config: PointConfig):
    xs, ys = points.transpose()
    ax.scatter(xs, ys, s=config.size, color=config.color)


default_point_configs = {
    'fixed': PointConfig(size=20, color='black'),
    'initial': PointConfig(size=20, color='red'),
    'moving': PointConfig(size=1, color='green')
}


def plot_game(
    ax,
    game: Game,
    point_configs: Dict[str, PointConfig] = default_point_configs,
):
    _plot_points(ax, game.fixed_points, point_configs["fixed"])
    # Plot the initial point
    _plot_points(ax, game.points[:1], point_configs["initial"])
    _plot_points(ax, game.points[1:], point_configs["moving"])

    # Set the axis parameters
    ax.set_title(f"ratio={game.ratio:.5}, rounds={game.rounds}")
    ax.set_aspect('equal')
    ax.set_axis_off()
