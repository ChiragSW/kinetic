# RL Post-training with verl

[verl](https://verl.readthedocs.io/en/latest/) is an RL post-training
framework for LLMs. It supports PPO, GRPO, DAPO, and related algorithms
on top of common training and rollout backends such as FSDP, Megatron,
vLLM, and SGLang.

This guide focuses only on the Kinetic parts of a verl run: building a
compatible GPU image, submitting a detached job, passing credentials,
staging input data, and keeping checkpoints durable. For algorithm
choices, reward design, and model-specific tuning, use the upstream verl
documentation.

:::{note}
verl RL jobs are CUDA/GPU workloads, not TPU workloads. Use Kinetic GPU
accelerators such as `gpu-h100`, `gpu-h100x8`, or `gpu-a100x8`.
:::

## Prerequisites

Before starting, you need:

- A Kinetic cluster provisioned with `kinetic up`.
- A GPU node pool for the size of run you want:

  ```bash
  kinetic pool add --accelerator gpu-h100 --project your-project-id
  ```

- An Artifact Registry or Docker Hub repository where Kinetic can push a
  prebuilt GPU image.
- Optional Hugging Face and Weights & Biases credentials in your local
  environment:

  ```bash
  export HF_TOKEN="hf_..."
  export WANDB_API_KEY="..."
  ```

The example below adapts verl's
[GSM8K PPO quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html).
It is a smoke-run template, not a recommendation about which algorithm
or reward to use in production.

## Build a Kinetic-compatible verl Image

verl's dependency stack is large and tightly coupled to CUDA, PyTorch,
and the rollout backend. Use Kinetic prebuilt mode and start from one of
the official verl Docker images instead of asking Kinetic to resolve
those packages from a plain `requirements.txt`.

Create `Dockerfile.verl` next to your launcher:

```dockerfile
FROM verlai/verl:vllm011.latest

# Kinetic prebuilt mode installs project requirements at pod startup.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# The published verl images carry the heavy CUDA/runtime dependencies.
# Install verl itself editable so examples, preprocessors, and trainer
# entrypoints are available inside the remote job.
ARG VERL_REF=main
RUN git clone https://github.com/verl-project/verl.git /opt/verl && \
    cd /opt/verl && \
    git checkout "${VERL_REF}" && \
    pip3 install --no-deps -e . && \
    pip3 install google-cloud-storage cloudpickle absl-py

WORKDIR /app
COPY remote_runner.py /app/remote_runner.py

ENV PYTHONUNBUFFERED=1
ENV VLLM_USE_V1=1
CMD ["python3"]
```

Build and publish it as the GPU prebuilt image for this Kinetic project:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export KINETIC_VERL_REPO="us-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/kinetic-verl"

gcloud artifacts repositories create kinetic-verl \
  --repository-format=docker \
  --location=us \
  --project="${GOOGLE_CLOUD_PROJECT}"

kinetic build-base \
  --repo "${KINETIC_VERL_REPO}" \
  --category gpu \
  --dockerfile ./Dockerfile.verl \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --yes
```

For reproducible experiments, pin `VERL_REF` to a release tag or commit
and pin the `verlai/verl` image tag you validated.

## Submit a verl Smoke Run

Use `@kinetic.submit()` for RL runs. verl launches Ray workers inside the
pod, and the outer Kinetic job remains the unit you monitor, clean up,
and reattach to.

Create `verl_gsm8k_ppo.py`:

```python
import kinetic


VERL_BASE_REPO = "us-docker.pkg.dev/your-project-id/kinetic-verl"


def _upload_directory_to_gcs(local_dir: str, gcs_dir: str) -> None:
    from pathlib import Path
    from google.cloud import storage
    from google.cloud.storage import transfer_manager

    if not gcs_dir.startswith("gs://"):
        raise ValueError(f"Expected a gs:// output path, got {gcs_dir!r}")

    bucket_name, _, prefix = gcs_dir[5:].partition("/")
    prefix = prefix.strip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_root = Path(local_dir)
    files = [
        str(path.relative_to(local_root))
        for path in local_root.rglob("*")
        if path.is_file()
    ]

    if files:
        transfer_manager.upload_many_from_filenames(
            bucket,
            files,
            source_directory=local_dir,
            blob_name_prefix=f"{prefix}/" if prefix else "",
            worker_type=transfer_manager.THREAD,
        )


@kinetic.submit(
    accelerator="gpu-h100",
    container_image="prebuilt",
    base_image_repo=VERL_BASE_REPO,
    capture_env_vars=["HF_TOKEN", "WANDB_*"],
)
def run_verl_gsm8k_ppo(
    prepared_data_dir: str | None = None,
    train_max_samples: int = 128,
    val_max_samples: int = 128,
    total_epochs: int = 1,
    resume_from: str | None = None,
):
    import os
    import shutil
    import subprocess
    from pathlib import Path

    verl_dir = Path("/opt/verl")
    data_dir = Path("/tmp/verl-data/gsm8k")
    checkpoint_root = Path("/tmp/verl-checkpoints")
    experiment_dir = checkpoint_root / "kinetic-verl" / "gsm8k-ppo"
    output_dir = os.environ["KINETIC_OUTPUT_DIR"].rstrip("/")

    if resume_from is not None:
        shutil.copytree(resume_from, experiment_dir, dirs_exist_ok=True)

    if prepared_data_dir is not None:
        data_dir = Path(prepared_data_dir)
    else:
        subprocess.run(
            [
                "python3",
                "examples/data_preprocess/gsm8k.py",
                "--local_save_dir",
                str(data_dir),
            ],
            cwd=verl_dir,
            check=True,
        )

    command = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={data_dir / 'train.parquet'}",
        f"data.val_files={data_dir / 'test.parquet'}",
        f"data.train_max_samples={train_max_samples}",
        f"data.val_max_samples={val_max_samples}",
        "data.train_batch_size=16",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "critic.model.path=Qwen/Qwen2.5-0.5B-Instruct",
        "critic.optim.lr=1e-5",
        "critic.ppo_micro_batch_size_per_gpu=1",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.project_name=kinetic-verl",
        "trainer.experiment_name=gsm8k-ppo",
        "trainer.logger=console",
        "trainer.val_before_train=False",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        "trainer.save_freq=1",
        "trainer.test_freq=5",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.default_local_dir={experiment_dir}",
        "trainer.default_hdfs_dir=null",
        "trainer.resume_mode=auto",
    ]

    subprocess.run(command, cwd=verl_dir, check=True)

    gcs_checkpoints = f"{output_dir}/checkpoints"
    _upload_directory_to_gcs(str(checkpoint_root), gcs_checkpoints)
    return {"checkpoints": gcs_checkpoints}


if __name__ == "__main__":
    job = run_verl_gsm8k_ppo()
    print(f"Submitted Kinetic job: {job.job_id}")
    print(job.result())
```

The defaults intentionally run on a small sample. Once the image,
credentials, model download, Ray startup, and checkpoint upload all work,
raise `train_max_samples`, `val_max_samples`, `trainer.total_epochs`,
and the batch sizes for the real experiment.

To use a dataset you already prepared, pass it with `kinetic.Data(...)`
at the call site. Kinetic resolves it to a normal path inside the pod,
and the trainer still receives local parquet paths:

```python
job = run_verl_gsm8k_ppo(
    prepared_data_dir=kinetic.Data(
        "gs://your-bucket/verl-data/gsm8k/",
        fuse=True,
    ),
)
```

## Resume from a Kinetic Checkpoint

verl's FSDP checkpoints are a directory tree under
`trainer.default_local_dir`, with `latest_checkpointed_iteration.txt`
tracking the latest saved step. Kinetic makes previous GCS outputs
available to a new job through `kinetic.Data(...)`.

Pass the previous checkpoint prefix at the call site:

```python
job = run_verl_gsm8k_ppo(
    resume_from=kinetic.Data(
        "gs://your-bucket/outputs/previous-job/checkpoints/kinetic-verl/gsm8k-ppo",
        fuse=True,
    ),
)
```

The function copies that resolved path into the local checkpoint
directory before verl starts. With `trainer.resume_mode=auto`, verl
resumes from the latest checkpoint found there.

For long runs, upload checkpoints during training instead of only after
the trainer exits. The simplest pattern is a small background sync that
periodically copies completed `global_steps_*` directories from
`checkpoint_root` to `KINETIC_OUTPUT_DIR`.

## Kinetic-specific Scaling Knobs

Keep these settings aligned when you scale beyond the smoke run:

- `accelerator` and `trainer.n_gpus_per_node`: use `gpu-h100x8` with
  `trainer.n_gpus_per_node=8`, `gpu-a100x4` with
  `trainer.n_gpus_per_node=4`, and so on.
- `trainer.nnodes`: keep this at `1` for single-node Kinetic GPU jobs.
  Use upstream verl multi-node guidance before trying multi-node RL.
- `actor_rollout_ref.rollout.tensor_model_parallel_size`: increase this
  when the rollout model itself needs multiple GPUs.
- `actor_rollout_ref.rollout.gpu_memory_utilization`: lower this if vLLM
  competes with FSDP for memory on the same GPU set.
- `data.train_files` and `data.val_files`: use local paths inside the
  pod. For large prepared datasets, pass them with
  `kinetic.Data("gs://...", fuse=True)` rather than downloading them in
  the function.
- `trainer.default_local_dir`: keep it on a writable local filesystem,
  then copy durable artifacts to `KINETIC_OUTPUT_DIR`.
- `capture_env_vars`: pass only the credentials the job needs, usually
  `HF_TOKEN` and optionally `WANDB_*`.

## Monitor the Run

The launcher prints a Kinetic job ID:

```bash
kinetic jobs status JOB_ID --project your-project-id
kinetic jobs logs --follow JOB_ID --project your-project-id
```

verl emits trainer metrics to the console when `trainer.logger=console`.
If you switch to W&B, keep `capture_env_vars=["WANDB_*"]` and set the
verl logger override accordingly.

When the function returns, checkpoints are uploaded under:

```text
$KINETIC_OUTPUT_DIR/checkpoints
```

`KINETIC_OUTPUT_DIR` is a per-job GCS prefix created by Kinetic. See
[Checkpointing](../guides/checkpointing.md) for retention and cleanup
details.

## Related pages

- [PyTorch Training](pytorch_training.md) - basic Kinetic GPU usage.
- [Container Images](../guides/containers.md) - custom prebuilt images
  and the `kinetic build-base` contract.
- [Detached Jobs](../guides/async_jobs.md) - monitor long-running
  `@kinetic.submit()` workloads.
- [Checkpointing](../guides/checkpointing.md) - durable outputs via
  `KINETIC_OUTPUT_DIR`.
