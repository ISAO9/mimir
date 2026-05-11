"""
00_environment_check.py
=======================

Sanity-check that the MIMIR environment is installed correctly:

    1. import each submodule and report version
    2. resolve the compute device (CUDA / MPS / CPU) with a banner
    3. run a tiny forward pass through NeuralVelocityField + ray tracing
       to confirm autograd works end-to-end

Run this first after `uv pip install -e .`. If anything fails, the rest of
the pipeline will not work.

Usage
-----
    python scripts/00_environment_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import mimir
from mimir.fields import NeuralVelocityField
from mimir.fields.neural_velocity_field import NVFConfig
from mimir.physics import batched_straight_ray_travel_time
from mimir.utils import resolve_device, set_global_seed
from mimir.utils.device import banner


def main() -> int:
    print(f"[mimir] version = {mimir.__version__}")
    print(f"[python] {sys.version.splitlines()[0]}")
    print(f"[torch] {torch.__version__}")
    print(f"[numpy] {np.__version__}")

    # 1. Reproducibility
    set_global_seed(20260507)

    # 2. Device
    device = resolve_device("auto")
    print(banner(device))

    # 3. End-to-end smoke test
    cfg = NVFConfig(domain_x=(0.0, 10.0), domain_z=(0.0, 10.0))
    field = NeuralVelocityField(cfg).to(device)

    # Two arbitrary rays
    sources = torch.tensor([[0.5, 5.0], [5.0, 0.5]], device=device)
    receivers = torch.tensor([[9.5, 5.0], [5.0, 9.5]], device=device)

    tt = batched_straight_ray_travel_time(field, sources, receivers, n_samples=64)
    loss = (tt ** 2).sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in field.parameters())

    print("[smoke] travel times (s):", tt.detach().cpu().numpy().tolist())
    print(f"[smoke] gradient flow: {'OK' if has_grad else 'BROKEN'}")
    if not has_grad:
        return 1

    print("[ok] MIMIR environment is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
