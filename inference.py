"""
Inference Script — TICE (Tumor Immune Control Environment)
=========================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name for the environment.

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

This script runs 3 tasks (easy, medium, hard). Each task is a single multi-step episode:
reset() → repeatedly: LLM picks (B-cell action, T-cell action) → step() → log → done

Final score per task = average reward across steps in that episode.
Overall score = average across all 3 tasks.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tice import TICEAction, TICEEnv
    from tice.models import B_CELL_ACTIONS, T_CELL_ACTIONS
except (ImportError, ModuleNotFoundError):
    from client import TICEEnv
    from models import B_CELL_ACTIONS, TICEAction, T_CELL_ACTIONS


# Load .env before reading env vars
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Config (match judging expectations) ---
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "tice"

TEMPERATURE = float(os.getenv("TICE_LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("TICE_LLM_MAX_TOKENS", "500"))

SUCCESS_SCORE_THRESHOLD = float(os.getenv("TICE_SUCCESS_SCORE_THRESHOLD", "0.0"))

TASKS: List[Tuple[str, str, str]] = [
    ("easy", "immune_cold", "easy"),
    ("medium", "immune_hot", "medium"),
    ("hard", "high_mutation", "hard"),
]


SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You control a tumor immune therapy simulator. On each turn you must choose exactly one
    B-cell action and one T-cell action.

    Valid B-cell actions:
    {", ".join(B_CELL_ACTIONS)}

    Valid T-cell actions:
    {", ".join(T_CELL_ACTIONS)}

    Objective:
    - Reduce and eradicate the tumor before timeout.
    - Preserve energy and avoid excessive B-cell and T-cell fatigue.
    - B cells improve detection. T cells do the damage.
    - In early phase, overcommitting T cells before reliable detection is usually wasteful.
    - If T-cell fatigue is high, recovery may be better than aggression.

    You must reply with a valid JSON object and nothing else:
    {{
      "b_cell_action": "<one valid B-cell action>",
      "t_cell_action": "<one valid T-cell action>",
      "reasoning": "<brief reasoning>"
    }}

    Rules:
    - Use only the valid action strings.
    - Base decisions only on the provided observation.
    - Keep reasoning short.
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by hackathon judges
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP]  step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def heuristic_action(observation) -> tuple[str, str]:
    if observation.episode_phase == "early":
        return "INCREASE_HIGH", "REST"
    if observation.detection_signal < 0.4:
        return "INCREASE_LOW", "ATTACK_LOW"
    if observation.t_cell_fatigue > 0.6:
        return "MAINTAIN", "REST"
    if observation.tumor_trend == "increasing":
        return "MAINTAIN", "ATTACK_MEDIUM"
    return "MAINTAIN", "ATTACK_LOW"


def build_user_prompt(observation) -> str:
    return textwrap.dedent(
        f"""
        Current episode context:
        - archetype: {observation.archetype}
        - difficulty: {observation.difficulty}
        - timestep: {observation.timestep}
        - episode_phase: {observation.episode_phase}
        - tumor_trend: {observation.tumor_trend}
        - detection_signal: {observation.detection_signal}
        - t_cell_effectiveness: {observation.t_cell_effectiveness}
        - resource_level: {observation.resource_level}
        - b_cell_fatigue: {observation.b_cell_fatigue}
        - t_cell_fatigue: {observation.t_cell_fatigue}
        - recent_outcome: {observation.recent_outcome}
        - feedback: {observation.feedback}

        Choose the next B-cell and T-cell actions.
        Respond with JSON only.
        """
    ).strip()


def sanitize_json_response(raw_response: str) -> str:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return cleaned


def coerce_action(raw_action: Any, valid_actions: list[str], fallback: str) -> str:
    if not isinstance(raw_action, str):
        return fallback

    normalized = raw_action.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in valid_actions:
        return normalized

    for candidate in valid_actions:
        if normalized == candidate.upper():
            return candidate

    return fallback


def get_llm_action(client: OpenAI, observation) -> tuple[TICEAction, str]:
    fallback_b, fallback_t = heuristic_action(observation)
    raw_response = ""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_response = completion.choices[0].message.content or ""
        parsed = json.loads(sanitize_json_response(raw_response))

        b_action = coerce_action(
            parsed.get("b_cell_action"),
            B_CELL_ACTIONS,
            fallback_b,
        )
        t_action = coerce_action(
            parsed.get("t_cell_action"),
            T_CELL_ACTIONS,
            fallback_t,
        )
        reasoning = str(parsed.get("reasoning", "")).strip() or "no_reasoning"

        return TICEAction(b_cell_action=b_action, t_cell_action=t_action), reasoning
    except Exception as exc:
        fallback_action = TICEAction(
            b_cell_action=fallback_b,
            t_cell_action=fallback_t,
        )
        return fallback_action, f"fallback:{type(exc).__name__}"


def require_api_key() -> str:
    if API_KEY:
        return API_KEY
    raise RuntimeError(
        "Missing API key. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY before running inference_llm.py."
    )

def require_image_name() -> str:
    if IMAGE_NAME:
        return IMAGE_NAME
    raise RuntimeError(
        "Missing docker image name. Set LOCAL_IMAGE_NAME (or IMAGE_NAME) before running inference_llm.py."
    )


async def run_task(task: str, archetype: str, difficulty: str, client: OpenAI) -> float:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    env = await TICEEnv.from_docker_image(require_image_name())
    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(archetype=archetype, difficulty=difficulty)
        obs = result.observation

        while not obs.done:
            action, reasoning = get_llm_action(client, obs)
            result = await env.step(action)
            obs = result.observation

            reward = float(result.reward if result.reward is not None else 0.0)
            done = bool(result.done)

            steps += 1
            rewards.append(reward)

            action_summary = (
                f"{action.b_cell_action}|{action.t_cell_action}|"
                f"phase={obs.episode_phase}|trend={obs.tumor_trend}|note={reasoning[:40]}"
            )
            log_step(step=steps, action=action_summary, reward=reward, done=done, error=None)

        score = (sum(rewards) / len(rewards)) if rewards else 0.0
        score = round(float(score), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        err = str(e)[:80]
        if steps == 0:
            log_step(step=1, action="error", reward=0.0, done=True, error=err)
            rewards = [0.0]
            steps = 1
        score = (sum(rewards) / len(rewards)) if rewards else 0.0
        score = round(float(score), 4)
        success = False

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return float(score)


async def main() -> None:
    _ = require_image_name()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=require_api_key())

    task_scores: List[float] = []
    for task, archetype, difficulty in TASKS:
        score = await run_task(
            task=task,
            archetype=archetype,
            difficulty=difficulty,
            client=llm_client,
        )
        task_scores.append(score)

    overall = sum(task_scores) / len(task_scores) if task_scores else 0.0
    print(f"[DEBUG] overall_score={overall:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
