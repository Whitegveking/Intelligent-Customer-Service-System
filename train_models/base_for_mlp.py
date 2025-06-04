from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

class BaseModel(ABC):

    def __init__(
        self,
        model: Union[Sequential, BaseEstimator],
        trained: bool = False
    ) -> None:
        self.model = model
        self.trained = trained  # 模型是否已训练

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, name: str):
        """加载模型"""
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:

        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        # accuracy = self.model.score(x_test, y_test)
        print('Accuracy: %.3f\n' % accuracy)

        return accuracy
