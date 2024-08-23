
from ccdp.config.configuration import ConfigurationManager
from ccdp.components.data_transformation import DataTransformation
from ccdp.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Configuration for Data Transformation
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        
        # Initialize the DataTransformation component
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Perform the data transformation and save the results
        data_transformation.transform_and_save()
        
        logger.info("Data transformation pipeline completed successfully.")
