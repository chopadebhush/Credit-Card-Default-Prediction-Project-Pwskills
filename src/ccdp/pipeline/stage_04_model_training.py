from ccdp.config.configuration import ConfigurationManager
from ccdp.components.model_training import ModelTraining
from ccdp.logging import logger


class ModelTrainingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
         #Load the configuration for model training
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        
        #Initialize the ModelTraining component
        model_training = ModelTraining(config=model_training_config)
        
        #Run the model training process
        model_training.run_training()
        
        logger.info("Model training pipeline completed successfully.")