"""
Waste Classifier Trainer Package
"""

from .trainer import (
    AdaptiveAugmentationDataset,
    AddGaussianNoise,
    ResourceTracker,
    ModelFactory,
    Trainer,
    ExperimentManager,
    extract_dataset,
    analyze_dataset,
    get_default_augmentation_strategies,
)

__all__ = [
    "AdaptiveAugmentationDataset",
    "AddGaussianNoise",
    "ResourceTracker",
    "ModelFactory",
    "Trainer",
    "ExperimentManager",
    "extract_dataset",
    "analyze_dataset",
    "get_default_augmentation_strategies",
]
