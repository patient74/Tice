from __future__ import annotations

import numpy as np


INITIAL_TUMOR_SIZE = 0.3
INITIAL_STEALTH_LEVEL = 0.0
INITIAL_TIMESTEP = 0

TUMOR_SIZE_RANGE = (0.0, 1.0)
VISIBILITY_RANGE = (0.05, 0.95)
RESISTANCE_RANGE = (0.0, 1.0)
PDL1_SUPPRESSION_RANGE = (0.0, 0.9)
MUTATION_RATE_RANGE = (0.01, 0.5)
STEALTH_LEVEL_RANGE = (0.0, 0.5)

ESCALATION_START_TIMESTEP = 15
ESCALATION_INCREMENT = 0.01
GROWTH_PRESSURE_FACTOR = 0.5
GROWTH_DELTA_FACTOR = 0.1
MUTATION_DELTA_RANGE = (0.02, 0.06)
MUTATION_TYPES = ("STEALTH", "RESISTANCE", "GROWTH", "SUPPRESSION")


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


class Tumor:
    def __init__(self) -> None:
        self.reset(
            {
                "visibility": 0.5,
                "resistance": 0.3,
                "pdl1_suppression": 0.2,
                "mutation_rate": 0.1,
                "growth_rate": 0.05,
                "mutation_impact": 0.2,
            }
        )

    def reset(self, params: dict) -> None:
        self.tumor_size = _clip(INITIAL_TUMOR_SIZE, *TUMOR_SIZE_RANGE)
        self.visibility = _clip(float(params["visibility"]), *VISIBILITY_RANGE)
        self.resistance = _clip(float(params["resistance"]), *RESISTANCE_RANGE)
        self.pdl1_suppression = _clip(
            float(params["pdl1_suppression"]), *PDL1_SUPPRESSION_RANGE
        )
        self.mutation_rate = _clip(float(params["mutation_rate"]), *MUTATION_RATE_RANGE)
        self.growth_rate = float(np.clip(float(params["growth_rate"]), 0.0, 1.0))
        self.mutation_impact = float(np.clip(float(params["mutation_impact"]), 0.0, 1.0))
        self.stealth_level = _clip(INITIAL_STEALTH_LEVEL, *STEALTH_LEVEL_RANGE)
        self.timestep = int(INITIAL_TIMESTEP)

    def step(self, t_cell_pressure: float) -> float:
        growth_multiplier = float(
            np.clip(1.0 - (GROWTH_PRESSURE_FACTOR * t_cell_pressure), 0.0, 1.0)
        )
        self.tumor_size = _clip(
            self.tumor_size + (self.growth_rate * growth_multiplier),
            *TUMOR_SIZE_RANGE,
        )

        if self.timestep > ESCALATION_START_TIMESTEP:
            self.mutation_rate = _clip(
                self.mutation_rate + ESCALATION_INCREMENT,
                *MUTATION_RATE_RANGE,
            )

        if float(np.random.random()) < self.mutation_rate:
            self.apply_mutation()

        self.timestep = int(self.timestep + 1)
        return self.tumor_size

    def apply_mutation(self) -> None:
        mutation_type = str(np.random.choice(MUTATION_TYPES))
        delta = float(
            self.mutation_impact
            * np.random.uniform(MUTATION_DELTA_RANGE[0], MUTATION_DELTA_RANGE[1])
        )

        if mutation_type == "STEALTH":
            self.visibility = _clip(self.visibility - delta, *VISIBILITY_RANGE)
            self.stealth_level = _clip(self.stealth_level + delta, *STEALTH_LEVEL_RANGE)
        elif mutation_type == "RESISTANCE":
            self.resistance = _clip(self.resistance + delta, *RESISTANCE_RANGE)
        elif mutation_type == "GROWTH":
            self.growth_rate = float(
                np.clip(self.growth_rate + (delta * GROWTH_DELTA_FACTOR), 0.0, 1.0)
            )
        elif mutation_type == "SUPPRESSION":
            self.pdl1_suppression = _clip(
                self.pdl1_suppression + delta, *PDL1_SUPPRESSION_RANGE
            )

    def get_true_state(self) -> dict:
        return {
            "tumor_size": float(self.tumor_size),
            "visibility": float(self.visibility),
            "resistance": float(self.resistance),
            "pdl1_suppression": float(self.pdl1_suppression),
            "mutation_rate": float(self.mutation_rate),
            "growth_rate": float(self.growth_rate),
            "mutation_impact": float(self.mutation_impact),
            "stealth_level": float(self.stealth_level),
            "timestep": int(self.timestep),
        }

    def is_eradicated(self) -> bool:
        return self.tumor_size <= TUMOR_SIZE_RANGE[0]

    def is_escaped(self) -> bool:
        return self.tumor_size >= TUMOR_SIZE_RANGE[1]
