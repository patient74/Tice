from __future__ import annotations


TUMOR_REDUCTION_MULTIPLIER = 10.0
ERADICATION_BONUS = 50.0
ESCAPE_PENALTY = 50.0
T_CELL_FATIGUE_PENALTY = 0.5
B_CELL_FATIGUE_PENALTY = 0.3
TISSUE_DAMAGE_PENALTY = 2.0
EFFECTIVE_GAIN_MULTIPLIER = 20.0
RESOURCE_WASTE_PENALTY = 0.2
TIMESTEP_COST = 0.1


def compute_reward(
    prev_tumor_size: float,
    curr_tumor_size: float,
    t_cell_result: dict,
    b_cell_state: dict,
    t_cell_state: dict,
    is_eradicated: bool,
    is_escaped: bool,
) -> float:
    tumor_reduction = prev_tumor_size - curr_tumor_size
    # Dense shaping: most steps get learning signal via tumor reduction; terminal outcomes
    # then add a large bonus/penalty to push for decisive resolution.
    reward = tumor_reduction * TUMOR_REDUCTION_MULTIPLIER

    if is_eradicated:
        reward += ERADICATION_BONUS
    if is_escaped:
        reward -= ESCAPE_PENALTY

    reward -= t_cell_state["fatigue"] * T_CELL_FATIGUE_PENALTY
    reward -= b_cell_state["fatigue"] * B_CELL_FATIGUE_PENALTY
    reward -= t_cell_result["tissue_damage"] * TISSUE_DAMAGE_PENALTY

    # Waste penalty discourages "run everything at max" when it doesn't translate into progress.
    energy_spent = (1.0 - t_cell_state["energy"]) + (1.0 - b_cell_state["energy"])
    effective_gain = tumor_reduction * EFFECTIVE_GAIN_MULTIPLIER
    resource_waste = max(0.0, energy_spent - effective_gain)
    reward -= resource_waste * RESOURCE_WASTE_PENALTY

    reward -= TIMESTEP_COST
    return round(float(reward), 4)
