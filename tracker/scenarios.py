"""Shared Scenario definitions so every script tests on the same things."""
from dataclasses import replace

import numpy as np
from .simulate import RadarConfig,TargetConfig

BASE_RADAR = RadarConfig()
TARGET = TargetConfig()

SCENARIOS = {
    "straight": ("straight", BASE_RADAR),
    "turning": ("turn", BASE_RADAR),
    "maneuvers": ("random", BASE_RADAR),
    "noisy": ("random", replace(BASE_RADAR, sigma_r=150.0, sigma_theta=0.015)),
    "missed_30": ("random", replace(BASE_RADAR,p_detect=0.7))
}

Q_VAL = [0.1, 1.0, 10.0, 100.0]

def random_radar(rng) -> RadarConfig:
    """A radar with randomly drawn noise and miss rate, for training on varied conditions.

    The ranges cover every fixed scenario above. Noise is drawn log-uniformly so a 25 m radar
    is as likely as a 200 m one."""
    return RadarConfig(
        sigma_r=float(np.exp(rng.uniform(np.log(25.0), np.log(200.0)))),
        sigma_theta=float(np.exp(rng.uniform(np.log(0.0025), np.log(0.02)))),
        p_detect=1.0 if rng.random() < 0.4 else float(rng.uniform(0.5, 1.0)),
    )
