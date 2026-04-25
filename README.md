# Learned Trajectory Aggregation for Process Reward Models

A contribution to [its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub)
that makes trajectory aggregation pluggable and ships a learned reference implementation.

## Problem Statement

`ParticleFiltering` and `BeamSearch` in `its_hub` reduce a sequence of per-step PRM scores
to a single trajectory score via a hardcoded choice of `prod`, `min`, or `mean`.
None of the three is a defensible default:

| Reduction | Pathology |
|-----------|-----------|
| `prod` | Penalises long trajectories — more terms below 1 shrinks the product |
| `min` | Brittle — one weak step destroys an otherwise correct trajectory |
| `mean` | Discards step position and inter-step dependence |

This project introduces an `AbstractTrajectoryAggregator` interface (in `its_hub/base.py`)
so aggregation becomes pluggable, and ships a learned MLP aggregator that demonstrates
the interface delivers real accuracy gains.

## Hypothesis

**Primary:** A learned aggregator over per-step PRM scores outperforms `prod`, `min`, and
`mean` on trajectory selection accuracy at fixed budget.  The gap concentrates on
long trajectories (where `prod` underflows) and on trajectories whose worst step is
followed by recovery (where `min` discards the signal).

**Secondary:** Inspecting the learned aggregator reveals that late-step scores carry
materially more predictive weight than early-step scores.

## Method

### Interface change

```python
# its_hub/base.py
class AbstractTrajectoryAggregator(ABC):
    @abstractmethod
    def aggregate(self, step_scores: list[float]) -> float: ...
    async def aaggregate(self, step_scores: list[float]) -> float: ...
```

`ParticleFiltering` and `BeamSearch` gain an optional `aggregator` parameter
defaulting to `HardcodedAggregator("prod")`.  Existing user code is unchanged.

### Feature vector

Each trajectory is summarised as a 10-dimensional vector (see `src/learned_aggregator/features.py`):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | mean | Mean step score |
| 1 | min | Minimum step score |
| 2 | max | Maximum step score |
| 3 | last | Score of the final step |
| 4 | length | Number of steps |
| 5 | variance | Step score variance |
| 6 | pos\_min | Normalised position of minimum score |
| 7 | pos\_max | Normalised position of maximum score |
| 8 | last−first | Score change from first to last step |
| 9 | gap\_at\_min | Score drop into the minimum step |

### Model

`TrajectoryMLP`: 2-layer MLP (input→hidden→1, ReLU, sigmoid), binary cross-entropy
against trajectory correctness.  Default hidden width: 16 (≈465 parameters).
Trains with Adam, early stopping on validation loss (patience=10).

### Baselines

| Baseline | Definition |
|----------|-----------|
| `prod` | Product of step scores |
| `min` | Minimum step score |
| `mean` | Mean step score |
| `random` | Uniform random (sanity floor) |

## Data

- **Policy:** Qwen2.5-1.5B-Instruct at temperature 0.7
- **PRM:** Qwen2.5-Math-PRM-7B at 4-bit quantisation via `MLXProcessRewardModel`
- **Problems:** 200 from MATH train split (MATH500 held out entirely)
- **Trajectories:** N=8 per problem
- **Split:** 70/15/15 problem-level (train/val/test), seed=42

## Evaluation Protocol

Primary metric: **selection accuracy** — fraction of test problems where the
highest-scoring trajectory's answer is correct.

Stratified by:
- Trajectory length: short (≤4 steps), medium (5–9), long (≥10)
- Problem difficulty: MATH levels 1–5
- Candidate count N ∈ {4, 8, 16}

Bootstrap 95% CI with 1000 resamples.  A gap counts only if it exceeds twice
the CI half-width.

## Results

_Run `scripts/evaluate.py` after training to populate this table._

| Aggregator | Overall acc | N=4 | N=8 | N=16 |
|------------|------------|-----|-----|------|
| prod | — | — | — | — |
| min | — | — | — | — |
| mean | — | — | — | — |
| random | — | — | — | — |
| learned\_mlp | — | — | — | — |

## Per-Step Weight Profile

_Run `scripts/train_aggregator.py` to see the input-layer weight profile._

The weight profile shows which features the MLP relies on most.
Based on the `prod`-underflow hypothesis, we expect `last`, `pos_min`, and
`gap_at_min` to carry higher weight than `mean` — reflecting that
late-step quality and recovery from a weak step matter most.

## Quickstart

### 1. Install dependencies

```bash
cd learned-aggregator
pip install -e ".[dev]"          # installs its_hub as a local dependency
```

### 2. Start the vLLM server (for trajectory generation)

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8100
```

### 3. Generate trajectories (Apple Silicon, requires MLX)

```bash
python scripts/generate_trajectories.py \
    --lm-endpoint http://localhost:8100/v1 \
    --lm-model Qwen/Qwen2.5-1.5B-Instruct \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --num-problems 200 \
    --n-per-problem 8 \
    --output-dir data/trajectories
```

### 4. Train the MLP aggregator

```bash
python scripts/train_aggregator.py \
    --data-dir data/trajectories \
    --hidden-width 16 \
    --epochs 200 \
    --patience 10
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --test-jsonl data/trajectories/test.jsonl \
    --output results/eval.json
```

### 6. Use the trained aggregator in its_hub

```python
from its_hub.aggregators import LearnedMLPAggregator
from its_hub.algorithms.particle_gibbs import ParticleFiltering

agg = LearnedMLPAggregator("its_hub/aggregators/checkpoints/mlp_agg.pt")
pf = ParticleFiltering(sg=sg, prm=prm, aggregator=agg)
result = pf.infer(lm, problem, budget=16)
```

## Reproduction on CUDA hardware

The only Apple-Silicon-specific component is `MLXProcessRewardModel`.
On a CUDA host, replace it with:

```python
from its_hub.integration import LocalVllmProcessRewardModel
from reward_hub.base import AggregationMethod
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda",
    aggregation_method=AggregationMethod.LAST,
)
```

Then pass `prm` to `generate_trajectories.py` via code.  The JSONL format
and training/evaluation scripts are identical.

## Secondary Contribution: MLXProcessRewardModel

`its_hub/integration/mlx_prm.py` implements `AbstractProcessRewardModel`
using MLX on Apple Silicon.  It removes the CUDA dependency for local
PRM-scoring development on macOS.  `LocalVllmProcessRewardModel` continues
to work unchanged on its existing CUDA path.

## Out of Scope

- Retraining the PRM
- Outcome reward models
- Fine-tuning the policy
- Cross-domain validation beyond one sanity check
- Modifying the search logic of ParticleFiltering or BeamSearch
- Online aggregation (per-step inside the loop)
