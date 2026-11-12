"""Shared Scenario definitions so every script tests on the same things."""
from dataclasses import replace
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