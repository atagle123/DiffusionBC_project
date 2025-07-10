import numpy as np
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    def __init__(self, norm_params):
        self.norm_params = norm_params

    @abstractmethod
    def normalize(self, data, field):
        """
        Abstract method to normalize data. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def unnormalize(self, data, field):
        """
        Abstract method to unnormalize data. Must be implemented by subclasses.
        """
        pass

class DebugNormalizer(BaseNormalizer):
    def normalize(self, data, field):
        """
        Debug normalization: returns the data as is.
        """
        return data

    def unnormalize(self, data, field):
        """
        Debug unnormalization: returns the data as is.
        """
        return data

class GaussianNormalizer(BaseNormalizer):
    def normalize(self, data, field):
        """
        Gaussian normalization: (data - mean) / std
        """
        params = self.norm_params[field]
        return (data - params["mean"]) / params["std"]

    def unnormalize(self, data, field):
        """
        Gaussian unnormalization: (data * std) + mean
        """
        params = self.norm_params[field]
        return (data * params["std"]) + params["mean"]


class MinMaxNormalizer(BaseNormalizer):
    def normalize(self, data, field):
        """
        Min-Max normalization: (data - min) / (max - min)
        """
        params = self.norm_params[field]
        return (data - params["min"]) / (params["max"] - params["min"])

    def unnormalize(self, data, field):
        """
        Min-Max unnormalization: (data * (max - min)) + min
        """
        params = self.norm_params[field]
        return (data * (params["max"] - params["min"])) + params["min"]
