from ccdp.config.configuration import ConfigurationManager
from ccdp.components.model_evaluation import ModelEvaluation
from ccdp.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Configuration for Model Evaluation
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        
        # Initialize the ModelEvaluation component
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Load the model and evaluate
        model_evaluation.load_model()
        model_evaluation.evaluate()
        
        logger.info("Model evaluation pipeline completed successfully.")