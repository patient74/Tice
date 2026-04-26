from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TICEAction, TICEObservation
except (ImportError, ModuleNotFoundError):
    from models import TICEAction, TICEObservation


class TICEEnv(EnvClient[TICEAction, TICEObservation, State]):
    def _step_payload(self, action: TICEAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[TICEObservation]:
        obs_data = payload.get("observation", {})
        observation = TICEObservation(
            tumor_trend=obs_data.get("tumor_trend", "stable"),
            detection_signal=obs_data.get("detection_signal", 0.0),
            t_cell_effectiveness=obs_data.get("t_cell_effectiveness", "low"),
            resource_level=obs_data.get("resource_level", "moderate"),
            b_cell_fatigue=obs_data.get("b_cell_fatigue", 0.0),
            t_cell_fatigue=obs_data.get("t_cell_fatigue", 0.0),
            recent_outcome=obs_data.get("recent_outcome", "no_effect"),
            timestep=obs_data.get("timestep", 0),
            episode_phase=obs_data.get("episode_phase", "early"),
            archetype=obs_data.get("archetype", ""),
            difficulty=obs_data.get("difficulty", ""),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


TiceEnv = TICEEnv
