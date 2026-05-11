"""
mimir.baselines — classical inversion baselines for comparison with MIMIR.

These baselines exist purely so we can publish defensible "MIMIR vs classical"
comparisons. The implementation aims for textbook fidelity (Aster, Borchers,
Thurber, "Parameter Estimation and Inverse Problems", 3rd ed., 2018) rather
than novelty — any reviewer should look at the code and recognize a
standard, well-implemented classical method.
"""

from mimir.baselines.eikonal_tomography import (
    ClassicalEikonalTomography,
    EikonalTomographyConfig,
    EikonalTomographyState,
)

__all__ = [
    "ClassicalEikonalTomography",
    "EikonalTomographyConfig",
    "EikonalTomographyState",
]
