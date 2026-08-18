# Training image for GCP Batch. No CUDA base image is needed: the torch cu130
# wheels bundle their own CUDA runtime, and Batch installs the host driver and
# wires it into the container when installGpuDrivers is set.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# models and tokenized datasets go on the boot disk, never on the gcsfuse mount
ENV HF_HOME=/cache/hf

WORKDIR /app

# dependencies resolve from the lockfile alone, so this layer survives source edits
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev

ENTRYPOINT ["posttrain"]
