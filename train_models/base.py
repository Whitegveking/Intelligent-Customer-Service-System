from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

class BaseModel(ABC):
    """所有模型的基础类"""

    def __init__(
        self,
        model: Union[Sequential, BaseEstimator],
        trained: bool = False
    ) -> None:
        self.model = model
        self.trained = trained  # 模型是否已训练

    @abstractmethod
    def train(self) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        """预测音频的情感"""
        pass

    @abstractmethod
    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """预测音频的情感的置信概率"""
        pass

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        """保存模型"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, name: str):
        """加载模型"""
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        """搭建模型"""
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the model on test data and print accuracy.

        Args:
            x_test (np.ndarray): Test samples
            y_test (np.ndarray): True labels
        """
        predictions = self.predict(x_test)
        
        # Convert one-hot encoded y_test to class labels
        y_test_labels = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test_labels)
        print(f'Accuracy: {accuracy:.3f}')

        return accuracy
