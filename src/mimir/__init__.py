"""
MIMIR — Multi-physics Implicit Inversion via Implicit Representation.

A unified differentiable framework for geophysical inverse problems built
around coordinate-based Neural Fields with Fourier feature embeddings,
physics-informed losses, and conformal uncertainty quantification.

Subpackages
-----------
fields    : Neural representations (velocity field, source field, MT field)
physics   : Differentiable physics operators (ray tracing, eikonal, wave eq)
losses    : Physics-informed loss functions
data      : Synthetic benchmarks and acquisition geometry
training  : Training loops with best-checkpoint persistence
viz       : Paper-grade figure utilities (PDF, English, white background)
utils     : Reproducibility, configuration, device selection
"""

__version__ = "0.1.0"
