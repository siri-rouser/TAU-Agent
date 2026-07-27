# TAU-Agent: An Agentic Retrieval-Augmented Framework for Traffic Anomaly Understanding

Official implementation of **TAU-Agent**, our submission to the **AI City Challenge 2026**.

🏆 2nd Place on Track 3 (TAR)  
🏆 12th Place on Track 7 (FETV)  
🏆 5th Place on Track 8 (PSI-VQA)

---

## Overview

> Coming soon.

Brief introduction of TAU-Agent, the retrieval agent, perception tools, evidence retrieval pipeline, and VLM-based reasoning framework.

---

## Environment

### Option 1: Docker (recommended)

Build the image:

```bash
docker build -f docker/Dockerfile.aicity -t tau-agent:latest .
```

This installs all pinned dependencies (see [Dockerfile.aicity](docker/Dockerfile.aicity)) on top of the `vllm/vllm-openai:v0.11.0` base image, and performs editable installs of the [transformers](transformers), [trl](trl), and [qwen-vl-utils](qwen-vl-utils) packages vendored in this repo.

Run the container:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/TAU-Agent \
  -v /output:/output \
  -v /data:/data \
  --name tau_agent_run \
  tau-agent:latest /bin/bash
```

### Option 2: Local Environment

**Prerequisites:** Python 3.12 and an NVIDIA GPU with CUDA 12.8 drivers.

1. Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install all dependencies, including PyTorch/vLLM and the rest of the `vllm/vllm-openai:v0.11.0` base image stack (see [requirements.txt](requirements.txt)):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

3. Install the vendored packages in editable mode:

```bash
pip install -e transformers
pip install -e trl
pip install -e qwen-vl-utils
```
---

## Data Preparation

Place all source videos under `data/videos/{dataset_name}/xxx.mp4`, e.g.:

```text
data/videos/tar_test/v=-3nwOfm1Pdk_0-00_0-16.mp4
```

You should include data from the following datasets: Accident-Bench, barbados_traffic_challenge, FETV, HTV, PSI_VQA, so-tad, TAD, TAD-benchmark, tar_test, UCF-Crimes, and Vad-R1.

For details on the data and how to download it, see the [AI City Challenge 2026 Track 3 page](https://www.aicitychallenge.org/2026-track3/).

---

## Project Structure

```text
TAU-Agent/
├── configs/
├── datasets/
├── retrieval/
├── training/
├── inference/
├── evaluation/
├── scripts/
├── checkpoints/
├── assets/
└── README.md