# Running on GCP Batch

The Spark is the primary machine; Batch is for when it is busy, or to prove a run can
happen somewhere else. The infrastructure — checkpoint bucket, image repository, job
service account — is Terraform in the `gcp-infra` repo. What lives here is the `Dockerfile`
that builds the image and `deploy/smoke.yaml`, the job that uses it.

Build and push. Cloud Build rather than local Docker because the VM is x86 and a Mac is
not, and emulating a torch install is slow enough to notice. `cloudbuild.yaml` at the root
is picked up automatically:

    gcloud builds submit --region=us-central1 \
      --substitutions=_TAG=$(git rev-parse --short HEAD)

That tags the image both with the commit and as `latest`. Tagging with the commit is what
makes a checkpoint traceable back to the code that produced it, which matters as soon as
two runs are being compared. Omit the substitution for a throwaway build and it lands as
`dev`.

Submit the smoke run — job names must be unique, hence the timestamp:

    gcloud batch jobs submit smoke-$(date +%s) \
      --location=us-central1 --config=deploy/smoke.yaml

Watch it, and read what it printed:

    gcloud batch jobs describe JOB --location=us-central1 --format='value(status.state)'

    gcloud logging read "labels.job_uid=$(gcloud batch jobs describe JOB \
      --location=us-central1 --format='value(uid)')" --format='value(textPayload)'

Checkpoints land in `gs://bel-posttrain/runs/smoke`.

## What needs a rebuild

Only what is baked into the image. Everything the job passes on the command line is read at
submit time, so most experiments cost a resubmit rather than a build:

| Change                                      | Rebuild? |
| ------------------------------------------- | -------- |
| `src/`, `config/`, `pyproject.toml`         | Yes      |
| Hyperparameters, via `--set` in `commands`  | No       |
| Machine type, Spot, disk, retry policy      | No       |
| Which config file the job runs              | No       |

A rebuild after a source-only change reuses the cached dependency layer and takes a minute
or two. The first build, or any change to `pyproject.toml` or `uv.lock`, reinstalls several
GB of torch and takes upwards of thirteen.

`deploy/smoke.yaml` pins the four decisions worth knowing:

| Setting                            | Why                                                          |
| ---------------------------------- | ------------------------------------------------------------ |
| `g2-standard-8`, one L4            | 24GB is ample for a 1.7B model at bf16, and it is the cheapest current-generation GPU |
| `provisioningModel: SPOT`          | Roughly a fifth of on-demand, in exchange for preemption      |
| `gcs` volume at `/mnt/disks/ckpt`  | gcsfuse, so `out_dir` is an ordinary path to the trainer      |
| `RETRY_TASK` on exit code `50001`  | Batch's code for a preempted Spot VM                          |

The last two are what make the first two safe together. A preempted task restarts the
container from scratch, and `resume` picks up the newest *complete* checkpoint from the
bucket — the same check the README's [Resuming](../README.md#resuming) section describes,
doing exactly the job it was written for. A preemption mid-save leaves a truncated
checkpoint on GCS and the next attempt skips it.

`HF_HOME` points at the boot disk, not the mount. Tokenizing 939k rows through gcsfuse
would be much slower than doing it locally and is pure waste, since the cache is
reproducible. The tradeoff is that the cache dies with the VM, so a restarted run pays the
tokenization cost again.

No GPU quota increase is needed: `GPUS_ALL_REGIONS` is 3000 and us-central1 allows 16
preemptible L4s.
