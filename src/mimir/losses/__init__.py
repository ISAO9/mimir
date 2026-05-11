"""mimir.losses — physics-informed loss functions."""

from mimir.losses.physics_losses import (
    travel_time_data_loss,
    total_variation_2d,
)
from mimir.losses.tgv import (
    AuxFieldConfig,
    AuxiliaryVectorField,
    total_generalized_variation_2d,
    huber_total_variation_2d,
)

__all__ = [
    "travel_time_data_loss",
    "total_variation_2d",
    "AuxFieldConfig",
    "AuxiliaryVectorField",
    "total_generalized_variation_2d",
    "huber_total_variation_2d",
]
