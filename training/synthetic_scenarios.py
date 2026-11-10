"""Synthetic encounter scenarios in the same format as ``load_bc_scenario``.

The frozen BC navigator only needs each vessel's start state and goal; everything in
between is chosen by the network. So a scenario is just a set of start and goal points.
This module builds scenarios in which vessels sail straight routes that cross near the
middle of the map, with knobs for how many conflicts there are and how close they get:

    n_vessels      how many vessels (the DQN needs a fixed number per pool)
    spread_s       std of the vessels' arrival times at the crossing area, seconds.
                   Small = vessels arrive together = many simultaneous conflicts.
                   Large = vessels arrive one after another = few conflicts.
    offset_m       how far each route may miss the map centre sideways, metres.
                   Small = routes pass through the same point; large = they miss each other.
    meet_time_s    typical time at which vessels reach the crossing area.

Returns ``(trajectories, timestamps, vessel_ids)`` exactly like ``load_bc_scenario``:
each trajectory has columns x, y, speed, heading; the first row is the state after one
step and the last row is the goal. Vessel IDs encode the seed, so different seeds count as
different ship sets (``build_unique_envs`` skips scenarios whose ship set it has seen).

CAVEAT: BC was trained on 15 real journeys. Straight routes through the map centre may lie
outside what it learned, so always check with an idle run (``scan_difficulty.py``) that
the vessels still reach their goals before training on these scenarios.
"""

import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "bc_navigation" / "data_bc" / "manifest.json"


def map_size():
    manifest = json.loads(MANIFEST.read_text())
    return float(manifest["map_width_m"]), float(manifest["map_height_m"])


SOURCES = ("real", "synthetic", "mixed")
DEFAULT_SPEC = "400:1500,250:1000"  # "spread_s:offset_m" pairs; see the module docstring


def parse_spec(spec: str):
    """'400:1500,250:1000' -> [(400.0, 1500.0), (250.0, 1000.0)] (spread_s, offset_m)."""
    settings = []
    for part in str(spec).split(","):
        try:
            spread, offset = part.strip().split(":")
            settings.append((float(spread), float(offset)))
        except ValueError:
            raise ValueError(
                f"Bad synthetic spec {spec!r}: expected comma-separated SPREAD_S:OFFSET_M pairs, "
                "for example '400:1500,250:1000'."
            ) from None
    if not settings:
        raise ValueError("Synthetic spec is empty.")
    return settings


def uses_synthetic(source: str, seed: int) -> bool:
    """Does this scenario seed use a generated scenario? 'mixed' alternates: even seeds synthetic."""
    if source not in SOURCES:
        raise ValueError(f"scenario_source must be one of {SOURCES}, got {source!r}.")
    return source == "synthetic" or (source == "mixed" and seed % 2 == 0)


def setting_for(source: str, spec: str, seed: int):
    """(spread_s, offset_m) for this seed. The settings in `spec` take turns."""
    settings = parse_spec(spec)
    step = 2 if source == "mixed" else 1  # mixed uses only even seeds for synthetic
    return settings[(seed // step) % len(settings)]


def make_crossing_scenario(
    n_vessels: int,
    seed: int,
    *,
    spread_s: float = 70.0,
    offset_m: float = 300.0,
    meet_time_s: float = 420.0,
    speed_range=(3.5, 6.0),
    heading_jitter_rad: float = 0.1,
    dt: int = 10,
):
    width, height = map_size()
    centre = np.array([width / 2.0, height / 2.0])
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 2.0 * math.pi)
    max_radius = 0.9 * min(width, height) / 2.0

    trajectories, timestamps = [], []
    for i in range(n_vessels):
        angle = base + 2.0 * math.pi * i / n_vessels + rng.uniform(-0.25, 0.25)
        speed = float(rng.uniform(*speed_range))
        radius = min(speed * max(120.0, meet_time_s + rng.normal(0.0, spread_s)), max_radius)
        out_angle = angle + math.pi + rng.uniform(-heading_jitter_rad, heading_jitter_rad)

        # Shift this vessel's whole route sideways so routes need not meet at one point.
        side = np.array([-math.sin(angle), math.cos(angle)])
        pivot = centre + side * rng.uniform(-offset_m, offset_m)
        start = pivot + radius * np.array([math.cos(angle), math.sin(angle)])
        goal = pivot + radius * np.array([math.cos(out_angle), math.sin(out_angle)])
        start = np.clip(start, [50.0, 50.0], [width - 50.0, height - 50.0])
        goal = np.clip(goal, [50.0, 50.0], [width - 50.0, height - 50.0])

        distance = float(np.linalg.norm(goal - start))
        step = speed * dt
        count = int(math.ceil(distance / step))
        along = np.minimum(np.arange(count + 1) * step, distance)
        positions = start + along[:, None] * ((goal - start) / distance)
        delta = np.diff(positions, axis=0)
        trajectories.append(np.column_stack((
            positions[1:],
            np.linalg.norm(delta, axis=1) / dt,
            np.arctan2(delta[:, 1], delta[:, 0]),
        )))
        timestamps.append(np.arange(count, dtype=np.int64) * dt)

    vessel_ids = [10_000 + seed * 100 + i for i in range(n_vessels)]
    return trajectories, timestamps, vessel_ids


__all__ = [
    "DEFAULT_SPEC", "SOURCES", "make_crossing_scenario", "map_size", "parse_spec", "setting_for", "uses_synthetic",
]
