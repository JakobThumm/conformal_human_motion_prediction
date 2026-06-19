# Docker Setup (CUDA dev container)

Docker configuration for running the **conformal_human_motion_prediction** codebase with NVIDIA
CUDA GPU support. This image is **ROS2-free** — the real-time ROS2 integration lives in the
separate [`chmp_workspace`](https://github.com/JakobThumm/chmp_workspace) /
[`chmp_inference`](https://github.com/JakobThumm/chmp_inference) repos.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose
- NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA GPU with CUDA support (tuned for an RTX 5090 / Blackwell, sm_120)

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Quick start

```bash
cd docker
./build.sh          # build the CUDA 12.8 image
./run.sh            # start the container (detached)
./shell.sh          # open a shell (runs as your host user)

# --- inside the container, one time ---
bash docker/setup_env.sh        # creates the `unc` venv, installs the package + cu128 torch
```

The `unc` virtualenv is **auto-activated** in new shells (you should see `(unc)` in the prompt).
Activate manually with `source unc/bin/activate` if needed.

## What's included

- **Base:** NVIDIA CUDA 12.8.1 with cuDNN on Ubuntu 24.04 (recent `ptxas` for sm_120).
- **Python env:** a venv at `/workspace/unc` with the package installed editable plus the
  validated GPU stack (`jax[cuda12]`, cu128 `torch`/`torchvision`) — set up by `setup_env.sh`.
- **Dev tools:** build essentials, Python 3, git, vim, sudo.
- **User mapping:** a container user matching your host UID/GID, so mounted files stay yours.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set (the GPU is often shared).

## Usage

The codebase is mounted at `/workspace`, so host edits are reflected live.

```bash
./shell.sh    # (unc) auto-activated

# run the pipeline (see the repo README for the full command set)
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m conformal_human_motion_prediction.examples.eval_full_pipeline --enable_ood
```

Verify GPU inside the container:

```bash
nvidia-smi
python -c "import jax; print(jax.devices())"
python -c "import torch; print(torch.cuda.is_available())"
```

Stop the container:

```bash
cd docker && docker-compose down
```

## User permissions

The container runs as your host user (UID/GID detected by `build.sh`/`run.sh`), with passwordless
sudo. Files created under `/workspace` are owned by you — no permission conflicts between host and
container.

## Troubleshooting

- **GPU not detected:** ensure nvidia-docker2 is installed and restart Docker
  (`sudo systemctl restart docker`); re-check with the `nvidia-smi` command above.
- **`torch.cuda.is_available() == False`:** confirm the cu128 wheels were installed
  (`pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`); the default
  PyPI wheel is cu130 and fails on the sm_120 driver.
- **XLA aborts with `PTX version ... does not support target 'sm_120a'`:** the in-container `ptxas`
  is too old for jaxlib — prepend a newer `ptxas` to `PATH` (see the `_cuda_ptxas_shim` note in
  the repo `CLAUDE.md`).
- **Permission issues:** verify `id` inside the container matches your host `id`; rebuild with
  `docker-compose build --no-cache` if the user setup failed.

## File structure

```
docker/
├── Dockerfile           # CUDA 12.8.1 + cuDNN image (no ROS2)
├── docker-compose.yml   # GPU reservation, /workspace + models mounts, user env
├── entrypoint.sh        # UID/GID matching + venv-activation bashrc
├── setup_env.sh         # creates the `unc` venv and installs the package + GPU stack
├── build.sh / run.sh / shell.sh / stop.sh
└── README.md            # this file
```
