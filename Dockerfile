# syntax=docker/dockerfile:1

# This file passes all checks from Hadolint (https://github.com/hadolint/hadolint)
# Use the command `hadolint Dockerfile` to test
# Adding Hadolint to `pre-commit` is non-trivial, so the command must be run manually

FROM python:3.11.11-slim-bookworm AS base

WORKDIR /bot

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get --no-install-recommends -y install git=1:2.39.* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip==24.3.1

# Install required packages
COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-lock.txt .
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . -c requirements-lock.txt

# Copy remaining files
COPY LICENSE .
COPY README.md .
COPY src/ src/

# Re-install so `pip` stores all metadata properly
RUN pip install --no-cache-dir --no-deps -e .

# Script executors
ENTRYPOINT ["python", "-m", "chiron_utils.scripts.run_bot"]

LABEL org.opencontainers.image.source=https://github.com/ALLAN-DIP/chiron-utils
