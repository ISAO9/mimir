"""
mimir.utils.device
==================

Resolve the compute device once, consistently, with sensible auto-selection.

Order of preference when `device='auto'`:
    CUDA  >  MPS (Apple Silicon)  >  CPU

We expose a thin wrapper rather than scattering `torch.cuda.is_available()`
checks across the codebase. This also lets us print a single banner about
which accelerator is in use.
"""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """
    Return a torch.device matching `spec`.

    Parameters
    ----------
    spec : {"auto", "cuda", "mps", "cpu"}
        Preferred device. "auto" picks the best available.
    """
    spec = spec.lower()

    if spec == "cuda" or (spec == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
        # explicit cuda requested but unavailable -> fall through to error path
        if spec == "cuda":
            raise RuntimeError("CUDA requested but not available.")

    if spec == "mps" or (spec == "auto" and torch.backends.mps.is_available()):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if spec == "mps":
            raise RuntimeError("MPS requested but not available.")

    return torch.device("cpu")


def banner(device: torch.device) -> str:
    """One-line human-readable description of the active device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        return f"[device] CUDA — {name} (sm_{cap[0]}{cap[1]})"
    if device.type == "mps":
        return "[device] MPS — Apple Silicon (Metal)"
    return "[device] CPU"
