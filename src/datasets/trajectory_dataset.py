import numpy as np
import d4rl
import gym
import torch
from dataclasses import dataclass
from collections import namedtuple
from src.datasets.dataset_utils import sequence_dataset, EpisodeDataset
from dataclasses import dataclass, field
from typing import List, Dict

Batch = namedtuple("Batch", "trajectories conditions")


class TrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_entry,
        clip_actions_to_eps: bool = True,
        history_len: int = 4,
        horizon: int = 16,
        stride: int = 1,
        max_n_episodes: int = 10000,
        pad_val: float = 0.0,
    ):
        self.history_len = history_len
        self.horizon = horizon

        self.env = gym.make(env_entry)
        dataset_itr = sequence_dataset(self.env)

        dataset = EpisodeDataset()
        for i, episode in enumerate(dataset_itr):
            dataset.add_episode(
                episode_data=episode
            )  # TODO discard episodes lesser than horizon

        # TODO manage max episode length... and max n episodes
        dataset.preprocess(history_len=history_len, pad_val=pad_val)

        self.normalizer = (
            dataset.normalizer
        )  # Get normalizer from dataset after preprocessing, to use in evaluation

        lengths_list = dataset.episodes_length
        self.indices = self.make_indices(
            lengths_list, traj_len=history_len + horizon, stride=stride
        )

        self.dataset = dataset
        self.get_env_attributes()

    def make_indices(
        self,
        episodes_lengths: list,
        traj_len: int,
        max_traj_len: int = 1000,
        stride: int = 1,
    ) -> np.ndarray:
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        assume that the dataset is already padded
        traj_len = history_len+horizon
        """
        indices = []
        for i, path_length in enumerate(episodes_lengths):
            max_start = min(path_length, max_traj_len) - traj_len
            assert max_start >= 0  # TODO pad min...
            for start in range(0, max_start, stride):
                end = start + traj_len
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_env_attributes(self):
        """
        Function to to get the env and his attributes from the dataset_name

        """
        action_space = self.env.action_space
        observation_space = self.env.observation_space

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int, eps: float = 1e-4) -> Batch:
        path_ind, start, end = self.indices[idx]

        observations = self.dataset.get_episode(path_ind)["observations"][
            start:end
        ]  # TODO maybe join both arrays for faster indexing...
        actions = self.dataset.get_episode(path_ind)["actions"][start:end]

        full_array = np.concatenate([actions, observations], axis=-1)
        conditions = full_array[: self.history_len, :]
        trajectories = full_array[self.history_len :, :]

        return Batch(trajectories, conditions)
