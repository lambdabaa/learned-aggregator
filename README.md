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

## Trajectory Corpus Pipeline

The canonical data pipeline is expressed as an `sdg_hub` Flow at
`flows/trajectory_corpus.yaml`.  Five blocks run in sequence:

| # | Block | Role |
|---|-------|------|
| 1 | `PromptBuilderBlock` (`build_math_prompt`) | Wraps each problem in the Qwen step-by-step system prompt (`flows/prompts/math_system.yaml`) |
| 2 | `RowMultiplierBlock` (`fan_out_trajectories`) | Fans out each problem row to N=8 trajectory candidates |
| 3 | `LLMChatBlock` (`generate_trajectory`) | Generates each trajectory with the policy LLM (async, temperature=0.7) |
| 4 | `MLXProcessRewardScoreBlock` (`score_steps`) | Scores each reasoning step prefix with the PRM; writes `step_scores: list[float]` |
| 5 | `MathVerifyAnswerBlock` (`verify_answer`) | Extracts `\boxed{}` answer; labels `correct: bool` against `ground_truth` |

`scripts/generate_trajectories.py` is a thin runner: it imports the custom blocks
(triggering `BlockRegistry` registration), loads the YAML, injects model config,
applies a seed-pinned 70/15/15 problem-level split, and writes per-split JSONL files.

To run just the flow programmatically:

```python
import learned_aggregator.blocks  # registers custom blocks
from sdg_hub import Flow

flow = Flow.from_yaml("flows/trajectory_corpus.yaml")
flow.set_model_config(
    model="openai/Qwen/Qwen2.5-1.5B-Instruct",
    api_base="http://localhost:8100/v1",
    api_key="NO_API_KEY",
)
result_df = flow.generate(dataset_df, runtime_params={"score_steps": {"model_name": "Qwen/Qwen2.5-Math-PRM-7B"}})
```

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

The pipeline is defined in `flows/trajectory_corpus.yaml`.
`scripts/generate_trajectories.py` is a thin runner that loads the flow,
splits problems 70/15/15, and writes `train.jsonl`, `val.jsonl`, `test.jsonl`.

```bash
python scripts/generate_trajectories.py \
    --lm-endpoint http://localhost:8100/v1 \
    --lm-model Qwen/Qwen2.5-1.5B-Instruct \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --num-problems 200 \
    --output-dir data/trajectories
```

N=8 trajectories per problem is set in the flow YAML (`RowMultiplierBlock.num_samples`);
edit `flows/trajectory_corpus.yaml` to change it.

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

## Stretch: Cross-Policy Transfer via training_hub

_Optional — may be skipped on M4 Max due to CUDA requirements._

The core experiment evaluates an aggregator trained on trajectories from
policy A (Qwen2.5-1.5B-Instruct at temperature 0.7).  A stronger test is
cross-policy transfer: train on policy A, evaluate on policy B where policy
B is a LoRA fine-tune of the same base model.

**Why it matters:** An aggregator that generalises across policies has learned
the intrinsic geometry of PRM scores rather than a policy-specific artefact.
If accuracy holds across policies, the aggregator is worth upstream.

**Approach using training_hub:**

```bash
# Fine-tune policy B with LoRA
training_hub train \
    --base-model Qwen/Qwen2.5-1.5B \
    --dataset lighteval/MATH \
    --output-dir checkpoints/policy_b \
    --lora-rank 16

# Serve policy B and generate trajectories
vllm serve checkpoints/policy_b --port 8101
python scripts/generate_trajectories.py \
    --lm-endpoint http://localhost:8101/v1 \
    --lm-model policy_b \
    --output-dir data/trajectories_policy_b

# Evaluate with aggregator trained on policy A
python scripts/evaluate.py \
    --test-jsonl data/trajectories_policy_b/test.jsonl \
    --checkpoint checkpoints/mlp_agg.pt
```

**Caveats:**
- `training_hub` LoRA backends assume CUDA.  The LoRA step requires
  a CUDA host or a remote training cluster; the rest of the pipeline
  (PRM scoring via MLX, evaluation) runs locally on Apple Silicon.
- LoRA fine-tuning on MATH with default hyperparameters may not produce
  a meaningfully distinct policy in a short run.  Use at least 1000 steps.

## Out of Scope

- Retraining the PRM
- Outcome reward models
- Fine-tuning the policy
- Cross-domain validation beyond one sanity check
- Modifying the search logic of ParticleFiltering or BeamSearch
- Online aggregation (per-step inside the loop)
