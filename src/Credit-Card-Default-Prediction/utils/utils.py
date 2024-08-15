import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException

def save_object(file_path: str, obj) -> None:
    """
    Save an object to a file using dill for serialization.
    
    Args:
        file_path (str): Path where the object should be saved.
        obj: The object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Load an object from a file using dill.
    
    Args:
        file_path (str): Path from which the object should be loaded.
        
    Returns:
        The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Perform data preprocessing steps such as removing duplicates and handling missing values.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data.

    Returns:
        (pd.DataFrame, pd.DataFrame): Preprocessed training and testing data.
    """
    try:
        # Removing duplicates
        train_df.drop_duplicates(keep='first', inplace=True)
        test_df.drop_duplicates(keep='first', inplace=True)

        # Handling missing values (if needed, extend this as per your requirements)
        train_df.fillna(method='ffill', inplace=True)
        test_df.fillna(method='ffill', inplace=True)

        return train_df, test_df

    except Exception as e:
        raise CustomException(e, sys)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering such as creating new features or modifying existing ones.

    Args:
        df (pd.DataFrame): DataFrame to perform feature engineering on.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    try:
        df['TOTAL_BILL_AMT'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
        df['TOTAL_PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)

        columns_to_drop = ['ID', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                           'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        df.drop(columns=columns_to_drop, inplace=True)

        return df

    except Exception as e:
        raise CustomException(e, sys)
