from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RandomForestHyperparameters:
    n_estimators: list[int] = field(default_factory=list)
    max_depth: list[Optional[int]] = field(default_factory=list)
    min_samples_split: Optional[list[int]] = None
    min_samples_leaf: Optional[list[int]] = None

@dataclass
class XGBoostHyperparameters:
    n_estimators: list[int] = field(default_factory=list)
    learning_rate: Optional[list[float]] = None
    max_depth: list[Optional[int]] = field(default_factory=list)


@dataclass
class ModelParameters:
    RandomForest: RandomForestHyperparameters = field(
        default_factory=RandomForestHyperparameters
    )
    XGBoost: XGBoostHyperparameters = field(default_factory=XGBoostHyperparameters)


@dataclass
class Config:
    model_parameters: ModelParameters = field(default_factory=ModelParameters)
