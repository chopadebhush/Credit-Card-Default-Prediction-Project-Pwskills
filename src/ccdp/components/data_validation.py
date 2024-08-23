import os
import pandas as pd
from pathlib import Path
from ccdp.logging import logger
from ccdp.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        missing_files = []
        for file_name in self.config.ALL_REQUIRED_FILES:
            file_path = self.config.validation_dir / f"{file_name}.csv"
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return False
        
        logger.info(f"All required files are present: {self.config.ALL_REQUIRED_FILES}")
        return True
    
    def validate_schema(self, file_path: Path, expected_columns: list) -> bool:
        try:
            df = pd.read_csv(file_path)
            actual_columns = df.columns.tolist()
            if actual_columns == expected_columns:
                logger.info(f"Schema validation passed for {file_path}")
                return True
            else:
                logger.error(f"Schema validation failed for {file_path}. Expected {expected_columns}, got {actual_columns}")
                return False
        except Exception as e:
            logger.error(f"Error during schema validation for {file_path}: {e}")
            return False

    def validate_data(self) -> bool:
        # Check if all files exist
        validation_status = self.validate_all_files_exist()

        if validation_status:
            # Validate schema of each file if all files exist
            expected_columns = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", 
                                "PAY_SEPT", "PAY_AUG", "PAY_JUL", "PAY_JUN", "PAY_MAY", "PAY_APR",
                                "BILL_AMT_SEPT", "BILL_AMT_AUG", "BILL_AMT_JUL", "BILL_AMT_JUN", "BILL_AMT_MAY", "BILL_AMT_APR",
                                "PAY_AMT_SEPT", "PAY_AMT_AUG", "PAY_AMT_JUL", "PAY_AMT_JUN", "PAY_AMT_MAY", "PAY_AMT_APR",
                                "default"]

            for file_name in self.config.ALL_REQUIRED_FILES:
                file_path = self.config.validation_dir / f"{file_name}.csv"
                if not self.validate_schema(file_path, expected_columns):
                    validation_status = False
                    break

        # Ensure the artifacts/data_validation directory exists
        os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)

        # Save validation status to status.txt
        self.save_validation_status(validation_status)

        if validation_status:
            logger.info("Data validation passed for all files.")
        else:
            logger.error("Data validation failed.")
        
        return validation_status
    
    def save_validation_status(self, validation_status: bool):
        status_file_path = self.config.STATUS_FILE
        with open(status_file_path, "w") as status_file:
            status_file.write(f"Validation status: {validation_status}")
        logger.info(f"Validation status saved to {status_file_path}")