import numpy as np
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    def __init__(self, norm_params):
        self.norm_params = norm_params

    @abstractmethod
    def normalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Abstract method to normalize data. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def unnormalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Abstract method to unnormalize data. Must be implemented by subclasses.
        """
        pass


class DebugNormalizer(BaseNormalizer):
    def normalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Debug normalization: returns the data as is.
        """
        return data

    def unnormalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Debug unnormalization: returns the data as is.
        """
        return data


class GaussianNormalizer(BaseNormalizer):
    def normalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Gaussian normalization: (data - mean) / std
        """
        params = self.norm_params[field]
        return (data - params["mean"]) / params["std"]

    def unnormalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Gaussian unnormalization: (data * std) + mean
        """
        params = self.norm_params[field]
        return (data * params["std"]) + params["mean"]


class MinMaxNormalizer(BaseNormalizer):
    def normalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Min-Max normalization: (data - min) / (max - min)
        """
        params = self.norm_params[field]
        return (data - params["min"]) / (params["max"] - params["min"])

    def unnormalize(self, data: np.ndarray, field: str) -> np.ndarray:
        """
        Min-Max unnormalization: (data * (max - min)) + min
        """
        params = self.norm_params[field]
        return (data * (params["max"] - params["min"])) + params["min"]
