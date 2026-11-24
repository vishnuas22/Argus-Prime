# Argus-Prime serve image (TASK-A3).
#
# Multi-stage: build deps in a builder, slim runtime in the final image.
# Runs as a non-root user. The artifact bundle is NOT baked in — the entrypoint
# verifies/fetches it at start (ARGUS_ARTIFACTS_DIR volume or HF fetch).
#
# Build:  docker build -t argus-prime:latest .
# Run:    docker run -p 8420:8420 -e ARGUS_API_KEYS=... argus-prime:latest

# ---------- builder ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY deepfake/requirements.txt ./deepfake/requirements.txt
RUN pip install --prefix=/install -r deepfake/requirements.txt

# ---------- runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ARGUS_ROOT=/app \
    ARGUS_ARTIFACTS_DIR=/app/artifacts \
    ARGUS_PORT=8420

# Runtime system libs: OpenCV needs libGL/libglib; exiftool for the metadata
# channel; ffmpeg for the audio path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libexpat1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libjpeg62-turbo \
        libpng16-16 \
        libtiff6 \
        libwebp7 \
        libopenjp2-7 \
        libxcb1 \
        libx11-6 \
        ffmpeg \
        exiftool \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Non-root user. The artifacts volume must be writable by this uid.
RUN groupadd --gid 10001 argus \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin argus

WORKDIR /app
COPY --chown=argus:argus . /app

# Writable dir for the mutable verdict ledger (the artifacts bundle is read-only).
RUN mkdir -p /app/data && chown argus:argus /app/data

# The venv is not copied; deps come from the builder prefix.
USER argus

EXPOSE 8420

# Fail closed if the artifact bundle is missing, then serve.
ENTRYPOINT ["/app/docker/entrypoint.sh"]