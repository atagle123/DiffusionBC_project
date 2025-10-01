import numpy as np
import torch
from src.datasets.normalization import BaseNormalizer


class HistoryBuffer:
    def __init__(
        self,
        normalizer: BaseNormalizer,
        history_len: int,
        action_dim: int,
        obs_dim: int,
        batch_size: int,
        pad_value: float = 0.0,
    ):
        self.normalizer = normalizer
        self.history_len = history_len
        self.action_dim = action_dim
        self.latest_known_index = history_len  # or 0

        self.history = np.full(
            (batch_size, history_len, obs_dim + action_dim), pad_value, dtype=np.float32
        )  # fill with pad value

    def add_state(self, observations: np.ndarray):  # needs to handles batch size>1
        observations = self._expand_dims(
            observations
        )  # add at least 2d check dims B, 1, O
        observations = self.normalizer.normalize(observations, "observations")

        self.history[:, -1, self.action_dim :] = observations

    def add_action(self, actions: np.ndarray):
        """
        Adds actions to the history buffer and shifts the buffer to the left to make room for new actions.

        The history buffer maintains a sequence of past actions and observations. When a new action is added:
        - The entire buffer is shifted left by one position, discarding the oldest entry.
        - The new action is inserted at the last position in the buffer.
        - Actions are not normalized here, as normalization is handled elsewhere.
        - The typical usage pattern is: add_state -> add_action (with shift) -> repeat.
        """
        actions = self._expand_dims(actions)

        self.history[:, :-1] = self.history[:, 1:]  # shift history to the left

        self.history[:, -1, : self.action_dim] = actions

    def _expand_dims(self, arr: np.ndarray, min_dims: int = 2) -> np.ndarray:
        while arr.ndim < min_dims:
            arr = np.expand_dims(arr, axis=0)  # add dim at the front
        return arr

    def __call__(self, device: str = "cuda:0") -> torch.Tensor:
        return torch.tensor(self.history, dtype=torch.float32, device=device)
