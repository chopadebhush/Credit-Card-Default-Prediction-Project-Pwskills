import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from ccdp.logging import logger
from ccdp.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self):
        with open(self.config.model_file, "rb") as model_file:
            self.model = pickle.load(model_file)
        logger.info(f"Model loaded from {self.config.model_file}")

    def evaluate(self):
        #Load test dataset
        test_df = pd.read_csv(self.config.test_file)
        X_test = test_df.drop(columns=['default'])  #Assuming 'default' is the target column
        y_test = test_df['default']
        
        #Make predictions
        y_pred = self.model.predict(X_test)
        
        #Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        #Save metrics
        self.save_metrics(accuracy, report)
        
        logger.info(f"Model evaluation completed with accuracy: {accuracy}")

    def save_metrics(self, accuracy, report):
        with open(self.config.evaluation_metrics_file, "w") as metrics_file:
            metrics_file.write(f"Accuracy: {accuracy}\n\n")
            metrics_file.write("Classification Report:\n")
            metrics_file.write(report)
        logger.info(f"Evaluation metrics saved to {self.config.evaluation_metrics_file}")
