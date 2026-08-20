import os
import sys
from pathlib import Path

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utlis.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.classfication_metric import (
    get_classification_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
import mlflow.sklearn


# Keep training runs and the MLflow UI on the same project-local database,
# regardless of the directory from which Python is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Network Security Model Training")


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    def track_mlflow(
        self,
        best_model,
        classification_train_metric,
        classification_test_metric
    ):

        with mlflow.start_run(run_name="model-training"):

            mlflow.log_metrics({
                "train_f1_score": classification_train_metric.f1_score,
                "train_precision_score": classification_train_metric.precision_score,
                "train_recall_score": classification_train_metric.recall_score,
                "test_f1_score": classification_test_metric.f1_score,
                "test_precision_score": classification_test_metric.precision_score,
                "test_recall_score": classification_test_metric.recall_score,
            })

            try:
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model"
                )
            except PermissionError as error:
                # Preserve the completed metrics when Windows blocks MLflow's
                # temporary artifact-staging directory.
                mlflow.set_tag("model_artifact_status", "not_logged")
                mlflow.set_tag("model_artifact_error", str(error))
                logging.warning("MLflow model artifact was not logged: %s", error)


    def train_model(
        self,
        x_train,
        y_train,
        x_test,
        y_test
    ):

        try:

            models = {

                "Random Forest": RandomForestClassifier(
                    verbose=1
                ),

                "Decision Tree": DecisionTreeClassifier(),

                "Gradient Boosting": GradientBoostingClassifier(
                    verbose=1
                ),

                "Logistic Regression": LogisticRegression(
                    verbose=1,
                    max_iter=1000
                ),

                "AdaBoost": AdaBoostClassifier()

            }


            params = {

                "Decision Tree": {

                    "criterion": [
                        "gini",
                        "entropy",
                        "log_loss"
                    ]

                },


                "Random Forest": {

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]

                },


                "Gradient Boosting": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],

                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9
                    ]

                },


                "Logistic Regression": {},


                "AdaBoost": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.5,
                        0.001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]

                }

            }


            # Evaluate all models
            model_report: dict = evaluate_models(

                x_train=x_train,
                y_train=y_train,

                x_test=x_test,
                y_test=y_test,

                models=models,
                params=params

            )


            # Get best model score
            best_model_score = max(
                model_report.values()
            )


            # Get best model name
            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(
                    best_model_score
                )
            ]


            # Get best model
            best_model = models[best_model_name]


            # Training prediction
            y_train_pred = best_model.predict(
                x_train
            )


            classification_train_metric = (
                get_classification_score(
                    y_true=y_train,
                    y_pred=y_train_pred
                )
            )


            # Testing prediction
            y_test_pred = best_model.predict(
                x_test
            )


            classification_test_metric = (
                get_classification_score(
                    y_true=y_test,
                    y_pred=y_test_pred
                )
            )


            # Track the selected model and both evaluation datasets in one run.
            self.track_mlflow(
                best_model,
                classification_train_metric,
                classification_test_metric
            )


            # Load preprocessor
            preprocessor = load_object(

                file_path=
                self.data_transformation_artifact
                .transformed_object_file_path

            )


            # Create model directory
            model_dir_path = os.path.dirname(

                self.model_trainer_config
                .trained_model_file_path

            )


            os.makedirs(
                model_dir_path,
                exist_ok=True
            )


            # Create NetworkModel object
            network_model = NetworkModel(

                preprocessor=preprocessor,
                model=best_model

            )


            # Save trained model
            save_object(

                file_path=
                self.model_trainer_config
                .trained_model_file_path,

                obj=network_model

            )


            # Create ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(

                trained_model_file_path=
                self.model_trainer_config
                .trained_model_file_path,

                train_metric_artifact=
                classification_train_metric,

                test_metric_artifact=
                classification_test_metric

            )


            logging.info(

                f"Model trainer artifact: "
                f"{model_trainer_artifact}"

            )


            return model_trainer_artifact


        except Exception as e:

            raise NetworkSecurityException(
                e,
                sys
            ) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            train_file_path = (

                self.data_transformation_artifact
                .transformed_train_file_path

            )


            test_file_path = (

                self.data_transformation_artifact
                .transformed_test_file_path

            )


            # Loading training array
            train_arr = load_numpy_array_data(
                train_file_path
            )


            # Loading testing array
            test_arr = load_numpy_array_data(
                test_file_path
            )


            # Splitting training data
            x_train = train_arr[:, :-1]

            y_train = train_arr[:, -1]


            # Splitting testing data
            x_test = test_arr[:, :-1]

            y_test = test_arr[:, -1]


            # Train models
            model_trainer_artifact = (
                self.train_model(

                    x_train=x_train,
                    y_train=y_train,

                    x_test=x_test,
                    y_test=y_test

                )
            )


            return model_trainer_artifact


        except Exception as e:

            raise NetworkSecurityException(
                e,
                sys
            ) from e
