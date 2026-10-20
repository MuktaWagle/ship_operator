"""Exact geometric features used by the frozen BC checkpoint."""

import numpy as np


MODEL_FEATURE_NAMES = [
    "x/map_width",
    "y/map_height",
    "cos_heading",
    "sin_heading",
    "speed/10",
    "goal_dx/map_width",
    "goal_dy/map_height",
    "goal_distance/map_diagonal",
    "cos_relative_goal_bearing",
    "sin_relative_goal_bearing",
]

MAP_WIDTH_M = 6670.351775572024
MAP_HEIGHT_M = 7783.644865119113
MAX_SPEED_MS = 10.0


def features(position, heading, speed, goal):
    """Build the seven pre-augmentation features used during training."""
    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return np.array([
        position[0] / MAP_WIDTH_M,
        position[1] / MAP_HEIGHT_M,
        np.cos(heading),
        np.sin(heading),
        speed / MAX_SPEED_MS,
        (goal[0] - position[0]) / MAP_WIDTH_M,
        (goal[1] - position[1]) / MAP_HEIGHT_M,
    ], dtype=np.float32)


def augment_features(x, manifest):
    """Add distance and relative-goal-bearing features exactly as trained."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError("Expected finite N x 7 original features.")

    width = float(manifest["map_width_m"])
    height = float(manifest["map_height_m"])
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("Invalid map dimensions.")

    dx = x[:, 5].astype(np.float64) * width
    dy = x[:, 6].astype(np.float64) * height
    distance = np.hypot(dx, dy)
    cosine = np.ones(len(x))
    sine = np.zeros(len(x))
    np.divide(
        dx * x[:, 2] + dy * x[:, 3],
        distance,
        out=cosine,
        where=distance > 0,
    )
    np.divide(
        dy * x[:, 2] - dx * x[:, 3],
        distance,
        out=sine,
        where=distance > 0,
    )
    return np.column_stack((
        x,
        distance / np.hypot(width, height),
        np.clip(cosine, -1, 1),
        np.clip(sine, -1, 1),
    )).astype(np.float32)
