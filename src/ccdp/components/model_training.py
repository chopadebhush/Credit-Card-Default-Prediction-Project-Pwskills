import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from ccdp.logging import logger
import pickle
from ccdp.entity import ModelTrainingConfig

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = RandomForestClassifier()

    def load_data(self) -> tuple:
        """
        Load the transformed training and testing data.
        """
        train_df = pd.read_csv(self.config.train_file)
        test_df = pd.read_csv(self.config.test_file)

        X_train = train_df.drop(columns=["default"])
        y_train = train_df["default"]
        
        X_test = test_df.drop(columns=["default"])
        y_test = test_df["default"]

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        """
        Train the model on the training data.
        """
        logger.info("Training the model...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on the test data.
        """
        logger.info("Evaluating the model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy}")
        logger.info(f"Classification report:\n{report}")

    def save_model(self):
        """
        Save the trained model to disk.
        """
        os.makedirs(os.path.dirname(self.config.model_file), exist_ok=True)
        with open(self.config.model_file, "wb") as model_file:
            pickle.dump(self.model, model_file)
        logger.info(f"Trained model saved to {self.config.model_file}")

    def run_training(self):
        """
        Complete the training pipeline: load data, train model, evaluate, and save the model.
        """
        X_train, y_train, X_test, y_test = self.load_data()
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.save_model()