"""
Waste Classifier Trainer Package
"""

from .trainer import (
    AdaptiveAugmentationDataset,
    AddGaussianNoise,
    RandomAffineWithReflectPad,
    ResourceTracker,
    ModelFactory,
    Trainer,
    ExperimentManager,
    FocalLoss,
    build_criterion,
    extract_dataset,
    analyze_dataset,
    get_advanced_stratification_labels,
    advanced_stratified_split,
    analyze_dataset_with_rich,
    print_dataset_structure_with_rich,
)

from .calibration import (
    get_logits_and_labels,
    fit_temperature,
    apply_reject_routing,
    find_optimal_threshold,
)

__all__ = [
    "AdaptiveAugmentationDataset",
    "AddGaussianNoise",
    "RandomAffineWithReflectPad",
    "ResourceTracker",
    "ModelFactory",
    "Trainer",
    "ExperimentManager",
    "FocalLoss",
    "build_criterion",
    "extract_dataset",
    "analyze_dataset",
    "get_advanced_stratification_labels",
    "advanced_stratified_split",
    "analyze_dataset_with_rich",
    "print_dataset_structure_with_rich",
    "get_logits_and_labels",
    "fit_temperature",
    "apply_reject_routing",
    "find_optimal_threshold",
]
