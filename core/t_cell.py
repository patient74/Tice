from __future__ import annotations

import numpy as np


INITIAL_ATTACK_STRENGTH = 0.8
INITIAL_ENERGY = 1.0
INITIAL_FATIGUE = 0.0
VALUE_RANGE = (0.0, 1.0)
FATIGUE_EFFECT_FACTOR = 0.8
TISSUE_DAMAGE_FACTOR = 0.3
EXHAUSTION_ON_THRESHOLD = 0.8
EXHAUSTION_OFF_THRESHOLD = 0.2

T_CELL_TRANSITIONS = {
    "ATTACK_HIGH": {"base": 0.18, "energy_delta": -0.22, "fatigue_delta": 0.14},
    "ATTACK_MEDIUM": {"base": 0.08, "energy_delta": -0.10, "fatigue_delta": 0.05},
    "ATTACK_LOW": {"base": 0.05, "energy_delta": -0.06, "fatigue_delta": 0.02},
    "REST": {"base": 0.00, "energy_delta": 0.16, "fatigue_delta": -0.15},
}

EXHAUSTED_ACTION_MAP = {
    "ATTACK_HIGH": "ATTACK_MEDIUM",
    "ATTACK_MEDIUM": "REST",
    "ATTACK_LOW": "REST",
    "REST": "REST",
}


def _clip_unit(value: float) -> float:
    return float(np.clip(value, *VALUE_RANGE))


class TCellAgent:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.attack_strength = _clip_unit(INITIAL_ATTACK_STRENGTH)
        self.energy = _clip_unit(INITIAL_ENERGY)
        self.fatigue = _clip_unit(INITIAL_FATIGUE)
        self.exhausted = False

    def step(
        self,
        action: str,
        b_detection: float,
        pdl1_suppression: float,
        tumor_resistance: float,
    ) -> dict:
        if self.exhausted:
            # Exhaustion is a hard constraint: once fatigued past a threshold, the environment
            # automatically downgrades aggression to enforce recovery behavior.
            action = EXHAUSTED_ACTION_MAP.get(action, "REST")

        transition = T_CELL_TRANSITIONS.get(action, T_CELL_TRANSITIONS["ATTACK_LOW"])
        base = float(transition["base"])
        self.energy = _clip_unit(self.energy + float(transition["energy_delta"]))
        self.fatigue = _clip_unit(self.fatigue + float(transition["fatigue_delta"]))

        # Multiplicative effective damage makes coordination real: weak detection, high suppression,
        # or high fatigue can collapse damage even when the agent chooses "ATTACK_HIGH".
        effective_damage = (
            base
            * float(np.clip(b_detection, *VALUE_RANGE))
            * (1.0 - float(np.clip(pdl1_suppression, *VALUE_RANGE)))
            * (1.0 - (self.fatigue * FATIGUE_EFFECT_FACTOR))
            * (1.0 - float(np.clip(tumor_resistance, *VALUE_RANGE)))
        )
        tissue_damage = base * self.fatigue * TISSUE_DAMAGE_FACTOR

        if self.fatigue >= EXHAUSTION_ON_THRESHOLD:
            self.exhausted = True
        if self.fatigue <= EXHAUSTION_OFF_THRESHOLD:
            self.exhausted = False

        self.attack_strength = _clip_unit(base)
        return {
            "effective_damage": _clip_unit(effective_damage),
            "tissue_damage": _clip_unit(tissue_damage),
            "base_damage": _clip_unit(base),
        }

    def get_state(self) -> dict:
        return {
            "attack_strength": float(self.attack_strength),
            "energy": float(self.energy),
            "fatigue": float(self.fatigue),
            "exhausted": bool(self.exhausted),
        }
