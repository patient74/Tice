from __future__ import annotations

import numpy as np

from .tcga_params import TCGA_PARAMS


DIFFICULTY_SCALES = {
    "easy": 0.6,
    "medium": 1.0,
    "hard": 1.5,
}

ARCHETYPES = ["immune_hot", "immune_cold", "high_mutation"]
DIFFICULTIES = ["easy", "medium", "hard"]
VISIBILITY_NOISE_STD = 0.05
SUPPRESSION_NOISE_STD = 0.05
VISIBILITY_MUTATION_COUNT_MAX = 5000.0


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _normalize(value: float, low: float, high: float) -> float:
    span = high - low
    if span <= 0:
        return 0.0
    return float(np.clip((value - low) / span, 0.0, 1.0))


def sample_tumor_params(archetype: str, difficulty: str) -> dict:
    if archetype not in TCGA_PARAMS:
        raise ValueError(f"Unknown archetype: {archetype}")
    if difficulty not in DIFFICULTY_SCALES:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # Difficulty scaling is intentionally simple: it amplifies the same underlying
    # cohort-shaped distributions rather than introducing a separate "hard mode" tumor.
    scale = DIFFICULTY_SCALES[difficulty]
    archetype_params = TCGA_PARAMS[archetype]

    # We sample cohort-backed variables first (TMB, mutation count, genomic instability),
    # then map them into environment-facing dynamics (mutation, visibility, suppression).
    tmb = float(
        np.random.lognormal(
            archetype_params["tmb"]["mean_log"],
            archetype_params["tmb"]["std_log"],
        )
    )
    mutation_count = float(
        np.random.lognormal(
            archetype_params["mutation_count"]["mean_log"],
            archetype_params["mutation_count"]["std_log"],
        )
    )
    genomic_instability = _clip(
        float(
            np.random.normal(
                archetype_params["genomic_instability"]["mean"],
                archetype_params["genomic_instability"]["std"],
            )
        ),
        0.0,
        1.0,
    )

    # Mutation rate is a bounded, monotonic function of TMB: higher burden → more adaptation pressure.
    mutation_rate = _clip(_sigmoid(tmb / 50.0) * 0.3 * scale, 0.01, 0.5)
    visibility = _clip(
        _normalize(mutation_count, 0.0, VISIBILITY_MUTATION_COUNT_MAX)
        + float(np.random.normal(0.0, VISIBILITY_NOISE_STD)),
        0.05,
        0.95,
    )
    # Suppression is modeled as a noisy proxy for immune evasion (PD‑L1-like effect).
    pdl1_suppression = _clip(
        genomic_instability + float(np.random.normal(0.0, SUPPRESSION_NOISE_STD)),
        0.0,
        0.9,
    )
    growth_rate = float(0.05 * scale)
    resistance = float(0.30 * scale)
    mutation_impact = float(genomic_instability)

    return {
        "archetype": archetype,
        "difficulty": difficulty,
        "tmb": tmb,
        "mutation_count": mutation_count,
        "genomic_instability": genomic_instability,
        "mutation_rate": mutation_rate,
        "visibility": visibility,
        "pdl1_suppression": pdl1_suppression,
        "growth_rate": growth_rate,
        "resistance": resistance,
        "mutation_impact": mutation_impact,
    }


def get_random_episode_params() -> dict:
    archetype = str(np.random.choice(ARCHETYPES))
    difficulty = str(np.random.choice(DIFFICULTIES))
    return sample_tumor_params(archetype, difficulty)
