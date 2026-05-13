# Tunix GRPO Example

[Tunix](https://tunix.readthedocs.io/) is a JAX-based LLM
post-training library with support for SFT, reinforcement learning, and
rollout integrations. This guide shows how to run a small Tunix GRPO
job through Kinetic while keeping the Kinetic-specific parts explicit:
TPU scheduling, credential capture, `kinetic.Data(...)` inputs, and
durable checkpoints.

The example is adapted from the upstream
[Tunix GRPO Gemma demo](https://tunix.readthedocs.io/en/latest/_collections/examples/grpo_gemma.html),
which trains Gemma on GSM8K-style math prompts. The Kinetic script uses
a tiny local JSONL dataset and Gemma 3 270M by default so that the job is
a smoke test, not a full RL experiment.

:::{note}
Tunix is JAX-native and is strongest on TPUs. Use a Kinetic TPU pool for
this guide. The bundled example uses `tpu-v5litepod-1`; scale to
`tpu-v5litepod-8`, `tpu-v6e-8`, or a larger topology once the smoke run
works.
:::

## What Kinetic Handles

Tunix owns the RL algorithm, model loading, LoRA application, rollout
configuration, reward functions, and training loop. Kinetic owns the
remote execution boundary:

- `@kinetic.submit()` schedules a detached TPU job.
- `capture_env_vars` forwards `HF_TOKEN` and optional `WANDB_*`
  credentials.
- `kinetic.Data(...)` sends the local or GCS dataset directory into the
  pod and resolves it to a normal filesystem path.
- `KINETIC_OUTPUT_DIR` gives Tunix/Orbax a durable checkpoint prefix.

## Prerequisites

- A Kinetic cluster provisioned with `kinetic up`.
- A TPU node pool for the example:

  ```bash
  kinetic pool add --accelerator tpu-v5litepod-1 --project your-project-id
  ```

- A Python environment that can import Kinetic locally.
- Remote dependencies available to the Kinetic job. For a scratch
  launcher directory, create a `requirements.txt` with the Tunix stack
  you have validated:

  ```text
  git+https://github.com/google/tunix
  git+https://github.com/google/qwix
  git+https://github.com/google/flax
  git+https://github.com/jax-ml/jax
  optax
  orbax-checkpoint
  grain
  huggingface_hub
  tensorboardX
  numpy>2
  ```

- Optional Hugging Face and Weights & Biases credentials:

  ```bash
  export HF_TOKEN="hf_..."
  export WANDB_API_KEY="..."
  ```

:::{tip}
Run this example from a small launcher directory when you are iterating
on Tunix dependencies. That keeps Kinetic's dependency discovery focused
on the packages needed for the remote job.
:::

## Data API Pattern

The important Kinetic pattern is that data is declared at the call site,
not downloaded inside the trainer by default:

```python
job = run_tunix_grpo(kinetic.Data("data/tunix_grpo_smoke"))
```

For a real dataset already staged in GCS, use the same function and
switch only the argument:

```python
job = run_tunix_grpo(
    kinetic.Data("gs://your-bucket/datasets/gsm8k-jsonl/", fuse=True),
    max_steps=100,
    train_limit=4096,
    eval_limit=256,
)
```

The remote function receives `data_dir` as a plain string path. The
example expects:

```text
data_dir/
├── train.jsonl
└── test.jsonl
```

Each row should contain `question` and `answer` fields. The bundled
smoke data uses GSM8K-style answers such as `"#### {answer}"`.

## Run the Example

```bash
python examples/tunix_grpo.py
```

The launcher writes a tiny local dataset, submits the remote job, prints
the Kinetic job ID, and waits for the result. From another terminal you
can monitor it with:

```bash
kinetic jobs status JOB_ID --project your-project-id
kinetic jobs logs --follow JOB_ID --project your-project-id
```

The runnable script is included below.

```{literalinclude} ../../examples/tunix_grpo.py
:language: python
```

## Checkpoints and Resuming

The example points Tunix's `RLTrainingConfig.checkpoint_root_directory`
at:

```text
$KINETIC_OUTPUT_DIR/checkpoints
```

`KINETIC_OUTPUT_DIR` is a per-job GCS prefix created by Kinetic. Tunix
uses Orbax checkpointing.
For production runs, set a stable `output_dir=` on the decorator or pass
one when you submit so a restarted job can reuse the same checkpoint
prefix:

```python
@kinetic.submit(
    accelerator="tpu-v5litepod-8",
    output_dir="gs://your-bucket/tunix-grpo/run-001",
    capture_env_vars=["HF_TOKEN", "WANDB_*"],
)
def run_tunix_grpo(data_dir: str, max_steps: int = 100):
    ...
```

## Scaling Notes

After the smoke run works, tune these together:

- `accelerator`: use a larger TPU topology before increasing model size.
- `model_id`: the script supports `google/gemma-3-270m-it` and
  `google/gemma-3-1b-it`; larger models need matching Tunix model setup.
- `mesh_shape`: the example derives it from `len(jax.devices())`.
  Review it when using nonstandard TPU slices.
- `max_steps`, `train_limit`, and `eval_limit`: the defaults prove the
  execution path only.
- `GRPOConfig.num_generations`: higher values improve the group-relative
  signal but increase rollout cost.
- `RolloutConfig.max_tokens_to_generate` and `max_prompt_length`: keep
  these aligned with your dataset and TPU memory.

## Related pages

- [Tunix SFT Example](tunix_sft.md) - supervised fine-tuning with Tunix.
- [Data](../guides/data.md) - `kinetic.Data(...)` for local and GCS
  inputs.
- [Checkpointing](../guides/checkpointing.md) - durable outputs through
  `KINETIC_OUTPUT_DIR`.
- [Detached Jobs](../guides/async_jobs.md) - long-running
  `@kinetic.submit()` workloads.
