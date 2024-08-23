
from ccdp.config.configuration import ConfigurationManager
from ccdp.components.data_validation import DataValidation
from ccdp.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        
        if not data_validation.validate_data():
            raise Exception("Data validation failed!")
        
        logger.info("Data validation succeeded.")