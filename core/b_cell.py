from __future__ import annotations

import numpy as np


INITIAL_DETECTION_LEVEL = 0.3
INITIAL_ENERGY = 1.0
INITIAL_FATIGUE = 0.0
VALUE_RANGE = (0.0, 1.0)
DETECTION_DECAY = 0.97
FATIGUE_GAIN_THRESHOLD = 0.7
FATIGUE_GAIN_MULTIPLIER = 0.5

B_CELL_TRANSITIONS = {
    "INCREASE_HIGH": {"detection_delta": 0.15, "energy_delta": -0.20, "fatigue_delta": 0.10},
    "INCREASE_LOW": {"detection_delta": 0.08, "energy_delta": -0.10, "fatigue_delta": 0.04},
    "MAINTAIN": {"detection_delta": 0.0, "energy_delta": -0.03, "fatigue_delta": 0.0},
    "REDUCE": {"detection_delta": -0.10, "energy_delta": 0.18, "fatigue_delta": -0.12},
}


def _clip_unit(value: float) -> float:
    return float(np.clip(value, *VALUE_RANGE))


class BCellAgent:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.detection_level = _clip_unit(INITIAL_DETECTION_LEVEL)
        self.energy = _clip_unit(INITIAL_ENERGY)
        self.fatigue = _clip_unit(INITIAL_FATIGUE)

    def step(self, action: str) -> None:
        transition = B_CELL_TRANSITIONS.get(action, B_CELL_TRANSITIONS["MAINTAIN"])
        detection_delta = float(transition["detection_delta"])

        if self.fatigue > FATIGUE_GAIN_THRESHOLD and detection_delta > 0.0:
            detection_delta *= FATIGUE_GAIN_MULTIPLIER

        self.detection_level = _clip_unit(self.detection_level * DETECTION_DECAY)
        self.detection_level = _clip_unit(self.detection_level + detection_delta)
        self.energy = _clip_unit(self.energy + float(transition["energy_delta"]))
        self.fatigue = _clip_unit(self.fatigue + float(transition["fatigue_delta"]))

    def get_state(self) -> dict:
        return {
            "detection_level": float(self.detection_level),
            "energy": float(self.energy),
            "fatigue": float(self.fatigue),
        }
