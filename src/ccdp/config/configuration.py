
from ccdp.constants import *
from ccdp.utils.common import read_yaml, create_directories
from ccdp.entity import (DataIngestionConfig,
                         DataValidationConfig,
                         DataTransformationConfig,
                         ModelTrainingConfig,
                         ModelEvaluationConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir],[config.validation_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            validation_dir=Path(config.validation_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.transformed_train_dir, config.transformed_test_dir])

        data_transformation_config = DataTransformationConfig(
            transformed_train_dir=Path(config.transformed_train_dir),
            transformed_test_dir=Path(config.transformed_test_dir),
            scaler_file=Path(config.scaler_file),
            train_file=Path(config.train_file),
            test_file=Path(config.test_file)
        )

        return data_transformation_config
    

    #Model Training Configuration
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.model_dir])

        model_training_config = ModelTrainingConfig(
            model_dir=Path(config.model_dir),
            model_file=Path(config.model_file),
            train_file=Path(config.train_file),
            test_file=Path(config.test_file)
        )

        return model_training_config
    

     # Model Evaluation Configuration
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        evaluation_metrics_path = Path(config.evaluation_metrics_file)
        create_directories([evaluation_metrics_path.parent])

        model_evaluation_config = ModelEvaluationConfig(
            model_file=Path(config.model_file),
            evaluation_metrics_file=evaluation_metrics_path,
            test_file=Path(config.test_file)
        )

        return model_evaluation_config



    
    