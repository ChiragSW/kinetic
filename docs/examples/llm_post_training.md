# LLM Post-training with SLIME

[SLIME](https://github.com/THUDM/slime) is an LLM post-training
framework for RL scaling. It connects Megatron for training with
SGLang for rollout generation, and its quickstart covers GRPO-style
training on NVIDIA GPUs.

This guide shows how to run the SLIME quickstart from Kinetic. The key
difference from a local SLIME run is the execution environment: SLIME
ships patched Megatron/SGLang dependencies in its Docker image, so use
that image as a Kinetic prebuilt GPU image and let Kinetic schedule,
stream logs, and persist outputs.

:::{note}
SLIME is a CUDA/GPU workflow, not a TPU workflow. Use Kinetic GPU
accelerators such as `gpu-h100x8` or `gpu-a100x8` for this guide.
The upstream SLIME quickstart is validated primarily on H-series
NVIDIA GPUs.
:::

## Prerequisites

Before starting, you need:

- A Kinetic cluster provisioned with `kinetic up`.
- An 8-GPU node pool for the quickstart. H100 is the recommended
  target:

  ```bash
  kinetic pool add --accelerator gpu-h100x8 --project your-project-id
  ```

- An Artifact Registry or Docker Hub repository where Kinetic can push
  the SLIME base image.
- Optional Hugging Face and Weights & Biases credentials in your local
  environment:

  ```bash
  export HF_TOKEN="hf_..."
  export WANDB_API_KEY="..."
  ```

The examples below use the GLM4-9B quickstart from the
[SLIME quickstart](https://thudm.github.io/slime/get_started/quick_start.html).
For production runs, review the upstream SLIME documentation for the
model-specific parallelism and reward settings.

## Build a Kinetic-compatible SLIME Image

Create a `Dockerfile.slime` next to your training launcher:

```dockerfile
FROM slimerl/slime:latest

# Kinetic prebuilt mode installs project requirements at pod startup.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

# Runtime dependencies for /app/remote_runner.py.
RUN uv pip install --system \
    absl-py \
    cloudpickle \
    google-cloud-storage

WORKDIR /app
COPY remote_runner.py /app/remote_runner.py

ENV PYTHONUNBUFFERED=1
CMD ["python3"]
```

Build and push it as the GPU prebuilt image for this Kinetic project:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export KINETIC_SLIME_REPO="us-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/kinetic-slime"

gcloud artifacts repositories create kinetic-slime \
  --repository-format=docker \
  --location=us \
  --project="${GOOGLE_CLOUD_PROJECT}"

kinetic build-base \
  --repo "${KINETIC_SLIME_REPO}" \
  --category gpu \
  --dockerfile ./Dockerfile.slime \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --yes
```

`kinetic build-base` bundles Kinetic's `remote_runner.py` into the build
context, which is why the Dockerfile can copy it into `/app`. After the
build, use this repository with `container_image="prebuilt"` and
`base_image_repo=...`.

## Submit the SLIME Quickstart

Use `@kinetic.submit()` for SLIME runs. RL post-training can run for
hours, so a detached job is easier to inspect and clean up than a
blocking `@kinetic.run()` call.

Create `slime_quickstart.py`:

```python
import os
from pathlib import Path

import kinetic


SLIME_BASE_REPO = "us-docker.pkg.dev/your-project-id/kinetic-slime"


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

    # Collect all file paths relative to the local root
    files = [
        str(p.relative_to(local_root))
        for p in local_root.rglob("*") if p.is_file()
    ]

    transfer_manager.upload_many_from_filenames(
        bucket,
        files,
        source_directory=local_dir,
        blob_name_prefix=f"{prefix}/" if prefix else "",
        worker_type=transfer_manager.THREAD,
    )


@kinetic.submit(
    accelerator="gpu-h100x8",
    container_image="prebuilt",
    base_image_repo=SLIME_BASE_REPO,
    capture_env_vars=["HF_TOKEN", "WANDB_*"],
)
def run_slime_glm4_quickstart(
    num_rollout: int = 2,
    resume_from: str | None = None,
):
    import os
    import shutil
    import subprocess
    from pathlib import Path

    output_dir = os.environ["KINETIC_OUTPUT_DIR"]
    local_save = Path("/tmp/slime-output/GLM-Z1-9B-0414_slime")
    local_ref = Path("/tmp/slime-output/GLM-Z1-9B-0414_torch_dist")
    local_save.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        shutil.copytree(resume_from, local_save, dirs_exist_ok=True)

    # Download the same model and datasets used by the upstream SLIME
    # quickstart. Keep them in /root because SLIME's example scripts use
    # those paths by default.
    setup = r"""
set -euxo pipefail
cd /root/slime

huggingface-cli download zai-org/GLM-Z1-9B-0414 \
  --local-dir /root/GLM-Z1-9B-0414
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

source scripts/models/glm4-9B.sh
if [ ! -d /tmp/slime-output/GLM-Z1-9B-0414_torch_dist ]; then
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /tmp/slime-output/GLM-Z1-9B-0414_torch_dist
fi
"""
    subprocess.run(["bash", "-lc", setup], check=True)

    # Patch the upstream convenience script for a short Kinetic smoke
    # run and for durable local staging before upload to GCS.
    script_path = Path("/root/slime/scripts/run-glm4-9B.sh")
    run_script = script_path.read_text()
    run_script = run_script.replace(
        "--ref-load /root/GLM-Z1-9B-0414_torch_dist",
        f"--ref-load {local_ref}",
    )
    run_script = run_script.replace(
        "--load /root/GLM-Z1-9B-0414_slime/",
        f"--load {local_save}/",
    )
    run_script = run_script.replace(
        "--save /root/GLM-Z1-9B-0414_slime/",
        f"--save {local_save}/",
    )
    run_script = run_script.replace(
        "--num-rollout 3000",
        f"--num-rollout {num_rollout}",
    )

    patched = Path("/tmp/run-glm4-9B-kinetic.sh")
    patched.write_text(run_script)
    patched.chmod(0o755)

    subprocess.run(["bash", str(patched)], check=True)

    gcs_save = f"{output_dir.rstrip('/')}/GLM-Z1-9B-0414_slime"
    _upload_directory_to_gcs(str(local_save), gcs_save)
    return {"checkpoints": gcs_save}


if __name__ == "__main__":
    job = run_slime_glm4_quickstart()
    print(f"Submitted Kinetic job: {job.job_id}")
    print(job.result())
```

The default `num_rollout=2` is a smoke run. Once the pod reaches the
training loop and writes checkpoints successfully, raise it to the
upstream value or your experiment target.

To resume from an existing checkpoint prefix, pass it through
`kinetic.Data(...)` at the call site. Kinetic resolves it to a regular
path inside the pod before `run_slime_glm4_quickstart()` starts:

```python
job = run_slime_glm4_quickstart(
    resume_from=kinetic.Data(
        "gs://your-bucket/previous-run/GLM-Z1-9B-0414_slime/",
        fuse=True,
    ),
)
```

:::{tip}
Run this launcher from a small directory that only contains the launcher
and `Dockerfile.slime`, or keep your `requirements.txt` minimal. The
SLIME image already owns the training stack, and extra project
requirements slow down prebuilt-mode startup.
:::

## Monitor the Run

The launcher prints the Kinetic job ID. From another terminal:

```bash
kinetic jobs status JOB_ID --project your-project-id
kinetic jobs logs --follow JOB_ID --project your-project-id
```

The SLIME script starts a local Ray head process inside the Kinetic pod,
then submits the Megatron/SGLang training job to that local Ray cluster.
The Kinetic job is still the outer unit of scheduling, logging, and
cleanup.

When the function returns, the wrapper uploads the staged SLIME
checkpoint directory to:

```text
$KINETIC_OUTPUT_DIR/GLM-Z1-9B-0414_slime
```

`KINETIC_OUTPUT_DIR` is a per-job GCS prefix created by Kinetic. See
[Checkpointing](checkpointing.md) for retention and cleanup details.

## Scale the Quickstart

The upstream GLM4-9B script splits one 8-GPU node into:

- 4 actor-training GPUs: `--actor-num-nodes 1` and
  `--actor-num-gpus-per-node 4`.
- 4 rollout GPUs: `--rollout-num-gpus 4`.
- SGLang engines with 2 GPUs each:
  `--rollout-num-gpus-per-engine 2`.

For a full run, increase `num_rollout`, tune the batch relationship,
and adjust the checkpoint interval:

```text
rollout-batch-size * n-samples-per-prompt = global-batch-size * num-steps-per-rollout
```

SLIME validates this relationship when `--num-steps-per-rollout` is set.
Keep `--use-dynamic-batch-size` enabled unless you have a reason
to pin micro-batches manually.

## Common Adjustments

- **Using a different supported model:** Change the sourced model config in
the SLIME script, the Hugging Face download target, and the conversion
paths together. SLIME model configs live under `/root/slime/scripts/models`.

- **Using Weights & Biases:** Uncomment `WANDB_ARGS` in the patched script
or maintain your own copy of the run script, then submit with
`capture_env_vars=["WANDB_*"]`.

- **Resume from a previous run:** Pass the previous GCS checkpoint prefix
with `kinetic.Data("gs://.../GLM-Z1-9B-0414_slime/", fuse=True)`, as
shown above. Megatron expects a writable filesystem checkpoint path, so
the wrapper copies the resolved `Data` path into `local_save` before
starting the training script.

- **Keep outputs durable during long runs:** The wrapper uploads after
SLIME exits. For multi-hour jobs, add a background sync process or
modify the run script to periodically copy completed checkpoint
directories to `KINETIC_OUTPUT_DIR`. `Data(...)` is for inputs that
exist before submission; it does not upload files created later inside
the pod.

## Related pages

- [PyTorch Training](pytorch_training.md) - basic Kinetic GPU usage.
- [Container Images](../advanced/containers.md) - custom prebuilt
  images and the `kinetic build-base` contract.
- [Detached Jobs](../advanced/async_jobs.md) - monitor long-running
  `@kinetic.submit()` workloads.
- [Checkpointing](checkpointing.md) - durable outputs via
  `KINETIC_OUTPUT_DIR`.
