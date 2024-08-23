import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from ccdp.logging import logger
import pickle
from ccdp.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.scaler = StandardScaler()

    def transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        # Identify numerical and categorical columns
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        # Apply standard scaling to numerical features
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        # Apply label encoding to categorical features
        for column in categorical_columns:
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column])
        
        return X

    def save_transformed_data(self, X: pd.DataFrame, y: pd.Series, file_name: str, is_train: bool = True):
        output_dir = self.config.transformed_train_dir if is_train else self.config.transformed_test_dir
        os.makedirs(output_dir, exist_ok=True)
        
        transformed_df = pd.concat([X, y], axis=1)
        file_path = output_dir / file_name
        transformed_df.to_csv(file_path, index=False)
        logger.info(f"Transformed data saved to {file_path}")

    def transform_and_save(self):
        # Load train and test datasets
        train_df = pd.read_csv(self.config.train_file)
        test_df = pd.read_csv(self.config.test_file)

        # Separate features and target
        X_train = train_df.drop(columns=["default"])
        y_train = train_df["default"]
        
        X_test = test_df.drop(columns=["default"])
        y_test = test_df["default"]
        
        # Transform train and test data
        logger.info("Starting data transformation for training data...")
        X_train_transformed = self.transform_data(X_train)
        
        logger.info("Starting data transformation for test data...")
        X_test_transformed = self.transform_data(X_test)
        
        # Save the transformed data
        self.save_transformed_data(X_train_transformed, y_train, "transformed_train.csv", is_train=True)
        self.save_transformed_data(X_test_transformed, y_test, "transformed_test.csv", is_train=False)

        # Save the scaler used during the transformation
        self.save_scaler()
        
    def save_scaler(self):
        os.makedirs(os.path.dirname(self.config.scaler_file), exist_ok=True)
        with open(self.config.scaler_file, "wb") as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        logger.info(f"Scaler saved to {self.config.scaler_file}")
