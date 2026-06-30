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
    FocalLoss,
    build_criterion,
    extract_dataset,
    analyze_dataset,
    get_default_augmentation_strategies,
    get_advanced_stratification_labels,
    advanced_stratified_split,
    analyze_dataset_with_rich,
    print_dataset_structure_with_rich,
)

__all__ = [
    "AdaptiveAugmentationDataset",
    "AddGaussianNoise",
    "ResourceTracker",
    "ModelFactory",
    "Trainer",
    "ExperimentManager",
    "FocalLoss",
    "build_criterion",
    "extract_dataset",
    "analyze_dataset",
    "get_default_augmentation_strategies",
    "get_advanced_stratification_labels",
    "advanced_stratified_split",
    "analyze_dataset_with_rich",
    "print_dataset_structure_with_rich",
]
