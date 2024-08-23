from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    validation_dir: Path
    STATUS_FILE: Path
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    transformed_train_dir: Path
    transformed_test_dir: Path
    scaler_file: Path
    train_file: Path
    test_file: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_dir: Path
    model_file: Path
    train_file: Path
    test_file: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_file: Path
    evaluation_metrics_file: Path
    test_file: Path