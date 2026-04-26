from openenv.core.env_server.types import Action, Observation
from pydantic import Field


B_CELL_ACTIONS = [
    "INCREASE_HIGH",
    "INCREASE_LOW",
    "MAINTAIN",
    "REDUCE",
]

T_CELL_ACTIONS = [
    "ATTACK_HIGH",
    "ATTACK_MEDIUM",
    "ATTACK_LOW",
    "REST",
]


class TICEAction(Action):
    b_cell_action: str = Field(
        ...,
        description=f"B cell command. One of: {B_CELL_ACTIONS}",
    )
    t_cell_action: str = Field(
        ...,
        description=f"T cell command. One of: {T_CELL_ACTIONS}",
    )


class TICEObservation(Observation):
    tumor_trend: str = Field(description="'increasing', 'stable', or 'decreasing'")
    detection_signal: float = Field(description="Noisy B cell detection level [0,1]")
    t_cell_effectiveness: str = Field(description="'high', 'medium', or 'low'")
    resource_level: str = Field(description="'abundant', 'moderate', or 'scarce'")
    b_cell_fatigue: float = Field(description="B cell fatigue [0,1]")
    t_cell_fatigue: float = Field(description="T cell fatigue [0,1]")
    recent_outcome: str = Field(
        description="'strong_response', 'weak_response', or 'no_effect'"
    )
    timestep: int = Field(description="Current timestep [0,50]")
    episode_phase: str = Field(
        description="'early' (t<15), 'mid' (15-35), or 'late' (t>35)"
    )
    archetype: str = Field(description="Tumor archetype for this episode")
    difficulty: str = Field(description="Episode difficulty tier")
    feedback: str = Field(default="", description="Step outcome description")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)


TiceAction = TICEAction
TiceObservation = TICEObservation
