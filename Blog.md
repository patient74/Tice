# TICE: a partially observable coordination environment for immune control

TICE represents a class of sequential decision problems where success depends on sustained strategy: reading weak signals, coordinating multiple subsystems, conserving resources, and adapting when the world changes mid-episode.

**TICE** (the Tumor Immune Control Environment) is a compact testbed for those capabilities.

TICE is not a medical simulator in the clinical sense. It is a reinforcement learning environment wrapped around a biologically inspired conflict: an evolving tumor versus an immune system that has to detect, track, and attack it before resources run out.

The twist is that the LLM never sees the true tumor state.

It only gets noisy, partial signals:

- is the tumor trend increasing or decreasing?
- how strong is the detection signal?
- are T cells currently effective?
- how tired are the B cells and T cells?
- are resources abundant or scarce?

From those signals alone, the model has to decide what to do next.

## Why this environment is hard

The policy in TICE controls two linked subsystems:

- **B cells** improve detection
- **T cells** do the actual damage

If the model attacks too early, T cells burn out before detection is high enough to matter.
If it waits too long, the tumor keeps growing and mutating.
If it ignores resource management, fatigue compounds and the policy collapses.

This makes TICE a compact but meaningful test of three things current LLMs still struggle with:

1. long-horizon planning
2. subsystem coordination
3. reasoning under hidden state

The environment also changes across episodes. Tumors are sampled from TCGA-inspired distributions derived from real cancer cohorts, producing three different archetypes:

- `immune_hot`
- `immune_cold`
- `high_mutation`

Concretely, the episode sampler draws:

- tumor mutational burden (TMB) and mutation count from lognormal distributions
- a genomic instability score from a normal distribution (clipped to \([0, 1]\))

with archetype-specific parameters inspired by:

- `immune_hot`: Skin Cutaneous Melanoma (SKCM), \(n=440\)
- `immune_cold`: Glioblastoma Multiforme (GBM), \(n=397\)
- `high_mutation`: top 25% TMB cohort from SKCM, \(n=110\)

Difficulty (`easy`, `medium`, `hard`) scales downstream dynamics (including growth/resistance and mutation rate), and additional observation-affecting quantities are derived from these draws (e.g., visibility from normalized mutation count with noise, and suppression from genomic instability with noise).

So the model cannot simply memorize a fixed script.

## What the agent actually does

We deliberately kept the action space learnable:

- 4 B-cell actions
- 4 T-cell actions
- 16 total joint action combinations per step

The concrete commands are:

- B cells: `INCREASE_HIGH`, `INCREASE_LOW`, `MAINTAIN`, `REDUCE`
- T cells: `ATTACK_HIGH`, `ATTACK_MEDIUM`, `ATTACK_LOW`, `REST`

That action simplification mattered. It reduced output complexity enough for a compact policy to learn useful control behavior with notebook-friendly training.

The reward function encourages real strategy rather than brute force:

- reward for shrinking the tumor
- large bonus for eradication
- large penalty for tumor escape
- penalties for fatigue, wasted energy, tissue damage, and time

In other words, TICE does not reward “always attack.”
It rewards timing.

The simplest way to see the coordination logic is to look at **effective damage**, which is (roughly) “attack strength × how well you can see the tumor × how suppressed you are”:

`effective_damage ≈ attack_base × detection × (1 − suppression)`

This one line explains why the environment is not solvable by brute force:

- if detection is low, big attacks still do little
- if suppression rises, you need to adapt (or rest/build detection) instead of repeating

The full implementation adds fatigue and resistance terms on top of this (see `core/t_cell.py`), but the core idea stays the same: **your attack only “counts” when the enabling signals are in place**.
## Training setup

We trained an instruction-tuned text model in three stages:

1. **Base model**
   No task-specific adaptation.

2. **SFT**
   Supervised fine-tuning on planner-generated demonstrations.

3. **GRPO**
   Reward-driven refinement using TICE reward tables.

The planner teacher acts as an oracle-style upper bound for this compact environment. The learned models are not expected to beat it immediately; the main question is whether a compact text policy can move from weak zero-shot behavior to coordinated control.

## What changed after training?

Here is the high-level result across 27 held-out evaluation episodes:

| Policy | Avg return | Win rate | Loss rate | Timeout rate | Avg final tumor |
|---|---:|---:|---:|---:|---:|
| Planner teacher | -9.37 | 29.6% | 0.0% | 70.4% | 0.612 |
| GRPO model | -31.52 | 25.9% | 55.6% | 18.5% | 0.676 |
| SFT model | -36.40 | 25.9% | 48.1% | 25.9% | 0.630 |
| Random | -64.96 | 0.0% | 77.8% | 22.2% | 0.927 |
| Base model | -73.54 | 3.7% | 63.0% | 33.3% | 0.852 |

The key story is the progression:

- **Base -> SFT**
  - return: `-73.54 -> -36.40`
  - win rate: `3.7% -> 25.9%`
  - final tumor size: `0.852 -> 0.630`

- **Base -> GRPO**
  - return: `-73.54 -> -31.52`
  - win rate: `3.7% -> 25.9%`
  - timeout rate: `33.3% -> 18.5%`
  - final tumor size: `0.852 -> 0.676`

- **SFT -> GRPO**
  - return improved by `+4.88`
  - timeout rate dropped by `7.4 percentage points`

So the main win here is not “the RL model beat the planner teacher.”
It did not.

The real result is that a compact text policy learned a materially better control policy than both:

- its own zero-shot baseline
- a random policy

in a partially observable, multi-step environment with interacting subsystems.

## What the figure shows

![TICE results](tice_results_story.png)

There are two useful ways to read this figure:

First, the average return improves sharply from the base model to SFT, and then improves again with GRPO. That means the model is not just learning the output format; it is learning better decisions.

Second, the outcome breakdown changes qualitatively. The base model almost never wins. After training, the model starts producing real wins, and GRPO reduces the fraction of unresolved timeout behavior compared with SFT.

That is the kind of evidence we wanted from this environment: not just a cleaner response format, but a visible shift in behavior.

## Why this matters

TICE is a useful environment for studying agent behavior in domains where:

- the full system state is hidden
- multiple subsystems must be coordinated
- immediate reward can be misleading
- adaptation matters over time

That structure shows up in far more places than oncology. Scientific decision-making, robotics, operations planning, and resource management all have this shape.

The biological framing makes the environment intuitive, but the underlying benchmark is general: can a compact policy learn to act strategically when simple reactive policies fail?

In TICE, the answer is yes.

Not perfectly. Not beyond the planner teacher. But clearly enough to see.

And that makes it a useful training ground for the next generation of agentic models.
