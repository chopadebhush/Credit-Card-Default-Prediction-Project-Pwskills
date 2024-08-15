import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def check_std_dev(self, train_df, test_df):
        try:
            train_df_datafeatures = [
                i for i in train_df.columns if train_df[i].dtypes in ['int64', 'float64']
            ]
            test_df_datafeatures = [
                i for i in test_df.columns if test_df[i].dtypes in ['int64', 'float64']
            ]
            
            for feature in train_df_datafeatures:
                if train_df[feature].std() == 0:
                    logging.info(f"Dropping feature '{feature}' from training data due to 0 standard deviation.")
                    train_df.drop(columns=feature, inplace=True)
            
            for feature in test_df_datafeatures:
                if test_df[feature].std() == 0:
                    logging.info(f"Dropping feature '{feature}' from testing data due to 0 standard deviation.")
                    test_df.drop(columns=feature, inplace=True)

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)

    def remove_duplicates(self, train_df, test_df):
        try:
            initial_train_rows = train_df.shape[0]
            initial_test_rows = test_df.shape[0]
            
            train_df.drop_duplicates(keep='first', inplace=True)
            test_df.drop_duplicates(keep='first', inplace=True)
            
            logging.info(f"Removed {initial_train_rows - train_df.shape[0]} duplicate rows from training data.")
            logging.info(f"Removed {initial_test_rows - test_df.shape[0]} duplicate rows from testing data.")

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def feature_transformation(self, train_df, test_df):
        try:
            train_df['TOTAL_BILL_AMT'] = train_df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
            train_df['TOTAL_PAY_AMT'] = train_df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)

            test_df['TOTAL_BILL_AMT'] = test_df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
            test_df['TOTAL_PAY_AMT'] = test_df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)

            columns_to_drop = ['ID', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                               'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            train_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            logging.info(f"Dropped columns: {columns_to_drop}")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT']
            categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Pipeline for numerical and categorical columns created successfully')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

def initiate_data_transformation(self, train_path, test_path):
    try:
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Read train and test data completed")

        # Print column names for debugging
        logging.info(f"Train Data Columns: {train_df.columns.tolist()}")
        logging.info(f"Test Data Columns: {test_df.columns.tolist()}")

        # Check and remove features with 0 standard deviation
        train_df, test_df = self.check_std_dev(train_df, test_df)
        logging.info('Removed features with 0 standard deviation')

        # Remove duplicate rows
        train_df, test_df = self.remove_duplicates(train_df, test_df)
        logging.info('Removed duplicate rows')

        # Apply feature transformations
        train_df, test_df = self.feature_transformation(train_df, test_df)
        logging.info('Feature transformation completed')

        # Get the preprocessor object
        preprocessing_obj = self.get_data_transformer_object()
        logging.info("Obtained preprocessing object")

        # Define target column name
        target_column_name = "CLASS_LABEL"

        # Check if the target column exists before dropping
        if target_column_name in train_df.columns:
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
        else:
            raise CustomException(f"Target column '{target_column_name}' not found in train DataFrame.", sys)

        if target_column_name in test_df.columns:
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
        else:
            raise CustomException(f"Target column '{target_column_name}' not found in test DataFrame.", sys)

        # Apply preprocessing object
        logging.info('Applying preprocessing object on training and testing data')
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        # Handle class imbalance
        smote_enn = SMOTEENN()
        input_feature_train_arr, target_feature_train_df = smote_enn.fit_resample(input_feature_train_arr, target_feature_train_df)
        input_feature_test_arr, target_feature_test_df = smote_enn.fit_resample(input_feature_test_arr, target_feature_test_df)

        # Combine features and targets
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        # Save the preprocessing object
        logging.info("Saving preprocessing object")
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )

    except Exception as e:
        raise CustomException(e, sys)
