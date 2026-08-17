import sys
import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS
)

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig

from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.logging.logger import logging

from networksecurity.utils.main_utlis.utils import (
    save_numpy_array_data,
    save_object
)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact: DataValidationArtifact = (
                data_validation_artifact
            )

            self.data_transformation_config: DataTransformationConfig = (
                data_transformation_config
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified
        in the training_pipeline.py file and returns a Pipeline object
        with the KNNImputer object as the first step.

        Args:
            self: DataTransformation

        Returns:
            A Pipeline object
        """

        logging.info(
            "Entered get_data_transformer_object method of "
            "DataTransformation class"
        )

        try:
            imputer: KNNImputer = KNNImputer(
                **DATA_TRANSFORMATION_IMPUTER_PARAMS
            )

            logging.info(
                f"Initialise KNNImputer with "
                f"{DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            processor: Pipeline = Pipeline(
                [
                    ("Imputer", imputer)
                ]
            )

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:

        logging.info(
            "Entered initiate_data_transformation method of "
            "DataTransformation class"
        )

        try:
            logging.info("Starting data transformation")

            # Read training data
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            # Read testing data
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # --------------------------------------------------
            # TRAINING DATA
            # --------------------------------------------------

            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_train_df = train_df[TARGET_COLUMN]

            target_feature_train_df = target_feature_train_df.replace(
                -1,
                0
            )

            # --------------------------------------------------
            # TESTING DATA
            # --------------------------------------------------

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_test_df = test_df[TARGET_COLUMN]

            target_feature_test_df = target_feature_test_df.replace(
                -1,
                0
            )

            # --------------------------------------------------
            # GET PREPROCESSOR
            # --------------------------------------------------

            preprocessor = self.get_data_transformer_object()

            # Fit preprocessor only on training data
            preprocessor_object = preprocessor.fit(
                input_feature_train_df
            )

            # Transform training data
            transformed_input_train_feature = (
                preprocessor_object.transform(
                    input_feature_train_df
                )
            )

            # Transform testing data
            transformed_input_test_feature = (
                preprocessor_object.transform(
                    input_feature_test_df
                )
            )

            # --------------------------------------------------
            # CREATE TRAIN AND TEST ARRAYS
            # --------------------------------------------------

            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_feature_test_df)
            ]

            # --------------------------------------------------
            # SAVE TRANSFORMED TRAIN DATA
            # --------------------------------------------------

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            # --------------------------------------------------
            # SAVE TRANSFORMED TEST DATA
            # --------------------------------------------------

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # --------------------------------------------------
            # SAVE PREPROCESSOR OBJECT
            # --------------------------------------------------

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )

            logging.info(
                "Data transformation completed successfully"
            )

            # --------------------------------------------------
            # CREATE DATA TRANSFORMATION ARTIFACT
            # --------------------------------------------------

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config.transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config.transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config.transformed_test_file_path
                )
            )

            logging.info(
                f"Data transformation artifact: "
                f"{data_transformation_artifact}"
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)