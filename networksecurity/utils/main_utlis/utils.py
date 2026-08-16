import os
import sys
import yaml

from networksecurity.exception.exception import NetworkSecurityException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(
    file_path: str,
    content: object,
    replace: bool = False
) -> None:

    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e