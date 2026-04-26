import os
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..core.b_cell import BCellAgent
    from ..core.reward import compute_reward
    from ..core.t_cell import TCellAgent
    from ..core.tumor import Tumor
    from ..data.sampler import get_random_episode_params, sample_tumor_params
    from ..models import B_CELL_ACTIONS, T_CELL_ACTIONS, TICEAction, TICEObservation
except ImportError:
    from core.b_cell import BCellAgent
    from core.reward import compute_reward
    from core.t_cell import TCellAgent
    from core.tumor import Tumor
    from data.sampler import get_random_episode_params, sample_tumor_params
    from models import B_CELL_ACTIONS, T_CELL_ACTIONS, TICEAction, TICEObservation


TREND_THRESHOLD = 0.02
DETECTION_NOISE_STD = 0.03
EARLY_PHASE_END = 15
MID_PHASE_END = 35
EFFECTIVENESS_HIGH_THRESHOLD = 0.6
EFFECTIVENESS_MEDIUM_THRESHOLD = 0.3
RESOURCE_ABUNDANT_THRESHOLD = 0.6
RESOURCE_MODERATE_THRESHOLD = 0.3
STRONG_RESPONSE_THRESHOLD = 0.08
WEAK_RESPONSE_THRESHOLD = 0.03
INITIAL_TUMOR_SIZE = 0.3
DEFAULT_LAST_T_RESULT = {
    "effective_damage": 0.0,
    "tissue_damage": 0.0,
    "base_damage": 0.0,
}


class TICEEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        archetype: str | None = None,
        difficulty: str | None = None,
        max_steps: int = 50,
    ):
        self.archetype = archetype or os.getenv("TICE_ARCHETYPE")
        self.difficulty = difficulty or os.getenv("TICE_DIFFICULTY")
        self.max_steps = int(max_steps)
        self.tumor = Tumor()
        self.b_cell = BCellAgent()
        self.t_cell = TCellAgent()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prev_tumor_size = INITIAL_TUMOR_SIZE
        self._last_t_result = dict(DEFAULT_LAST_T_RESULT)
        self._current_archetype = "immune_hot"
        self._current_difficulty = "medium"

    def reset(
        self,
        archetype: str | None = None,
        difficulty: str | None = None,
        **_: dict,
    ) -> TICEObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        selected_archetype = archetype or self.archetype
        selected_difficulty = difficulty or self.difficulty

        if selected_archetype and selected_difficulty:
            params = sample_tumor_params(selected_archetype, selected_difficulty)
        else:
            params = get_random_episode_params()

        self._current_archetype = params["archetype"]
        self._current_difficulty = params["difficulty"]
        self.tumor.reset(params)
        self.b_cell.reset()
        self.t_cell.reset()
        self._prev_tumor_size = INITIAL_TUMOR_SIZE
        self._last_t_result = dict(DEFAULT_LAST_T_RESULT)

        return self._make_observation(
            reward=0.0,
            feedback="Episode started.",
            done=False,
        )

    def step(self, action: TICEAction) -> TICEObservation:  # type: ignore[override]
        self._state.step_count += 1

        b_action = action.b_cell_action
        t_action = action.t_cell_action

        # The server is the source of truth; invalid client actions are coerced to safe defaults
        # so training/eval can continue without crashing on formatting errors.
        if b_action not in B_CELL_ACTIONS:
            b_action = "MAINTAIN"
        if t_action not in T_CELL_ACTIONS:
            t_action = "ATTACK_LOW"

        prev_size = float(self.tumor.tumor_size)
        t_cell_pressure = float(
            np.clip(
                (self.t_cell.fatigue * 0.3) + (self.b_cell.detection_level * 0.3),
                0.0,
                1.0,
            )
        )

        # Update order matters for partial observability: the tumor advances first, then the
        # immune subsystems act, so the agent is always reacting with a one-step lag.
        self.tumor.step(t_cell_pressure)

        t_result = self.t_cell.step(
            t_action,
            self.b_cell.detection_level,
            self.tumor.pdl1_suppression,
            self.tumor.resistance,
        )
        self.b_cell.step(b_action)
        self._last_t_result = t_result

        self.tumor.tumor_size = float(
            np.clip(self.tumor.tumor_size - t_result["effective_damage"], 0.0, 1.0)
        )
        self._prev_tumor_size = prev_size

        is_eradicated = self.tumor.is_eradicated()
        is_escaped = self.tumor.is_escaped()
        is_timeout = self.tumor.timestep >= self.max_steps
        done = bool(is_eradicated or is_escaped or is_timeout)

        reward = compute_reward(
            prev_size,
            self.tumor.tumor_size,
            t_result,
            self.b_cell.get_state(),
            self.t_cell.get_state(),
            is_eradicated,
            is_escaped,
        )

        if is_eradicated:
            feedback = "VICTORY: Tumor eradicated."
        elif is_escaped:
            feedback = "DEFEAT: Tumor escaped."
        elif is_timeout:
            feedback = "TIMEOUT: Episode limit reached."
        else:
            feedback = (
                f"Tumor: {self.tumor.tumor_size:.3f} | "
                f"Damage: {t_result['effective_damage']:.3f} | "
                f"Reward: {reward:+.3f}"
            )

        return self._make_observation(reward=reward, feedback=feedback, done=done)

    @property
    def state(self) -> State:
        return self._state

    def _make_observation(
        self,
        reward: float,
        feedback: str,
        done: bool,
    ) -> TICEObservation:
        delta = self.tumor.tumor_size - self._prev_tumor_size
        if delta > TREND_THRESHOLD:
            trend = "increasing"
        elif delta < -TREND_THRESHOLD:
            trend = "decreasing"
        else:
            trend = "stable"

        raw_detection = self.b_cell.detection_level * (1.0 - self.tumor.stealth_level)
        detection = float(
            np.clip(raw_detection + np.random.normal(0.0, DETECTION_NOISE_STD), 0.0, 1.0)
        )
        # Detection signal is intentionally noisy/ambiguous: a drop could mean weaker B cells
        # or increased tumor stealth. This forces inference instead of direct state access.

        effectiveness_score = (1.0 - self.tumor.pdl1_suppression) * (1.0 - self.t_cell.fatigue)
        if effectiveness_score > EFFECTIVENESS_HIGH_THRESHOLD:
            effectiveness = "high"
        elif effectiveness_score > EFFECTIVENESS_MEDIUM_THRESHOLD:
            effectiveness = "medium"
        else:
            effectiveness = "low"

        average_energy = (self.b_cell.energy + self.t_cell.energy) / 2.0
        if average_energy > RESOURCE_ABUNDANT_THRESHOLD:
            resource_level = "abundant"
        elif average_energy > RESOURCE_MODERATE_THRESHOLD:
            resource_level = "moderate"
        else:
            resource_level = "scarce"

        last_damage = self._last_t_result["effective_damage"]
        if last_damage > STRONG_RESPONSE_THRESHOLD:
            recent_outcome = "strong_response"
        elif last_damage > WEAK_RESPONSE_THRESHOLD:
            recent_outcome = "weak_response"
        else:
            recent_outcome = "no_effect"

        timestep = int(self.tumor.timestep)
        if timestep < EARLY_PHASE_END:
            episode_phase = "early"
        elif timestep < MID_PHASE_END:
            episode_phase = "mid"
        else:
            episode_phase = "late"

        return TICEObservation(
            tumor_trend=trend,
            detection_signal=round(detection, 2),
            t_cell_effectiveness=effectiveness,
            resource_level=resource_level,
            b_cell_fatigue=round(self.b_cell.fatigue, 2),
            t_cell_fatigue=round(self.t_cell.fatigue, 2),
            recent_outcome=recent_outcome,
            timestep=timestep,
            episode_phase=episode_phase,
            archetype=self._current_archetype,
            difficulty=self._current_difficulty,
            feedback=feedback,
            done=done,
            reward=reward,
        )


TiceEnvironment = TICEEnvironment
