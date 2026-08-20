import os
import sys
import pickle

import numpy as np
import yaml

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object) -> None:
    try:
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered save_object method")

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(
            f"Object saved successfully at: {file_path}"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        logging.info("Entered load_object method")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(
            f"Object loaded successfully from: {file_path}"
        )

        return obj

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        logging.info("Entered load_numpy_array_data method")

        array = np.load(file_path)

        logging.info(
            f"NumPy array loaded successfully from: {file_path}"
        )

        return array

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_numpy_array_data(
    file_path: str,
    array: np.ndarray
) -> None:
    try:
        logging.info("Entered save_numpy_array_data method")

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        np.save(
            file_path,
            array
        )

        logging.info(
            f"NumPy array saved successfully at: {file_path}"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(
    x_train,
    y_train,
    x_test,
    y_test,
    models,
    params
):
    try:
        report = {}

        for model_name, model in models.items():

            param = params.get(model_name, {})

            if param:

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param,
                    cv=3,
                    # Windows can reject the worker-process pipes used by
                    # joblib's parallel backend in this environment.
                    n_jobs=1
                )

                gs.fit(
                    x_train,
                    y_train
                )

                # Get the fitted best model
                model = gs.best_estimator_

            else:

                # Fit the model
                model.fit(
                    x_train,
                    y_train
                )

            # Predictions using fitted model
            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            # Training accuracy
            train_score = accuracy_score(
                y_train,
                y_train_pred
            )

            # Testing accuracy
            test_score = accuracy_score(
                y_test,
                y_test_pred
            )

            # Store test score
            report[model_name] = test_score

            # IMPORTANT:
            # Store the fitted model back into models
            models[model_name] = model

            logging.info(
                f"{model_name}: "
                f"train_score={train_score}, "
                f"test_score={test_score}"
            )

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
