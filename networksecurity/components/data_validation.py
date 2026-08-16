import os
import sys
import pandas as pd

from scipy.stats import ks_2samp

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataValidationConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.constant.training_pipeline import SCHEMA_FILE_NAME

from networksecurity.utils.main_utlis.utils import (
    read_yaml_file,
    write_yaml_file
)


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self.schema_config = read_yaml_file(
                SCHEMA_FILE_NAME
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_number_of_columns(
        self,
        dataframe: pd.DataFrame
    ) -> bool:

        try:
            number_of_columns = len(
                self.schema_config["columns"]
            )

            logging.info(
                f"Required number of columns: {number_of_columns}"
            )

            logging.info(
                f"Data frame has columns: {len(dataframe.columns)}"
            )

            if len(dataframe.columns) == number_of_columns:
                return True

            return False

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def detect_dataset_drift(
        self,
        base_df,
        current_df,
        threshold=0.05
    ) -> bool:

        try:
            status = True
            report = {}

            for column in base_df.columns:

                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)

                if threshold <= is_same_dist.pvalue:
                    is_found = False

                else:
                    is_found = True
                    status = False

                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })

            drift_report_file_path = (
                self.data_validation_config.drift_report_file_path
            )

            # Create directory
            dir_path = os.path.dirname(
                drift_report_file_path
            )

            os.makedirs(
                dir_path,
                exist_ok=True
            )

            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(
        self
    ) -> DataValidationArtifact:

        try:

            train_file_path = (
                self.data_ingestion_artifact.trained_file_path
            )

            test_file_path = (
                self.data_ingestion_artifact.testing_file_path
            )

            # Read train and test data
            train_dataframe = DataValidation.read_data(
                train_file_path
            )

            test_dataframe = DataValidation.read_data(
                test_file_path
            )

            # Validate number of columns
            train_status = self.validate_number_of_columns(
                dataframe=train_dataframe
            )

            test_status = self.validate_number_of_columns(
                dataframe=test_dataframe
            )

            validation_status = (
                train_status and test_status
            )

            # Check data drift
            drift_status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            validation_status = (
                validation_status and drift_status
            )

            # Create directory for valid files
            dir_path = os.path.dirname(
                self.data_validation_config.valid_train_file_path
            )

            os.makedirs(
                dir_path,
                exist_ok=True
            )

            # Save validated train data
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True
            )

            # Save validated test data
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True
            )

            # Create DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,

                valid_train_file_path=(
                    self.data_validation_config.valid_train_file_path
                ),

                valid_test_file_path=(
                    self.data_validation_config.valid_test_file_path
                ),

                invalid_train_file_path=None,

                invalid_test_file_path=None,

                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                )
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e