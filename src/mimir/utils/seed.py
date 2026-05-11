"""
mimir.utils.seed
================

Globally fix random seeds across Python's `random`, NumPy, and PyTorch
(both CPU and the appropriate accelerator) so experiments are reproducible.

For paper-grade statistics we run multi-seed bootstraps; this module is the
single point of truth for "set everything for seed s".
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set the global RNG seed for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The seed value. We use 7- or 8-digit numbers in MIMIR
        (e.g. 20260507) to make the date-of-experiment self-documenting.
    deterministic : bool, default True
        If True, request deterministic algorithms in PyTorch where
        possible. Slightly reduces speed but guarantees bitwise reproducibility
        for paper figures.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # MPS (Apple Silicon) does not currently support manual_seed_all but
    # respects torch.manual_seed above. No-op here for clarity.

    if deterministic:
        # See https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        # cuBLAS workspace config (required for some deterministic ops on CUDA >=10.2)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - best-effort, older torch fall-through
            pass
