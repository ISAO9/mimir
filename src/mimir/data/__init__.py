"""mimir.data — synthetic benchmarks and acquisition geometry."""

from mimir.data.benchmarks import (
    BenchmarkSpec,
    make_gaussian_anomaly,
    make_layered,
    make_curvefault_lookalike,
)
from mimir.data.acquisition import (
    AcquisitionGeometry,
    cross_well_layout,
    surface_layout,
    sample_observed_travel_times,
)

__all__ = [
    "BenchmarkSpec",
    "make_gaussian_anomaly",
    "make_layered",
    "make_curvefault_lookalike",
    "AcquisitionGeometry",
    "cross_well_layout",
    "surface_layout",
    "sample_observed_travel_times",
]
