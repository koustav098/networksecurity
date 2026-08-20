import sys

from networksecurity.exception.exception import NetworkSecurityException


class NetworkModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def predict(self, x):
        try:
            transformed_feature = self.preprocessor.transform(x)

            return self.model.predict(transformed_feature)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e