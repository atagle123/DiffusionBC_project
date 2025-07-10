import numpy as np
import d4rl
import gym
import torch
from dataclasses import dataclass
from collections import namedtuple

Batch = namedtuple("Batch", "states next_states actions rewards masks")


@dataclass
class Dataset:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    dones: np.ndarray


class D4RL_Dataset(torch.utils.data.Dataset):
    def __init__(self, env_entry, clip_actions_to_eps=True):

        self.env = gym.make(env_entry)
        dataset = d4rl.qlearning_dataset(self.env)

        self.normalizer = None

        if clip_actions_to_eps:
            eps = 1e-5
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        dataset["rewards"] = dataset["rewards"].reshape(-1, 1)
        dataset["terminals"] = dataset["terminals"].reshape(-1, 1)

        self.dataset = Dataset(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
        )

        self.norm_rewards(env_entry)

        self.get_env_attributes()

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

    def get_returns(self):
        episode_return = 0
        episode_returns = []

        assert len(self.dataset.rewards) == len(self.dataset.dones)
        assert len(self.dataset.observations) == len(self.dataset.dones)

        for i in range(len(self.dataset.rewards)):
            episode_return += self.dataset.rewards[i]

            if self.dataset.dones[i]:
                episode_returns.append(episode_return)
                episode_return = 0.0

        return episode_returns

    def norm_rewards(self, env_entry):

        if "antmaze" in env_entry:
            self.dataset.rewards -= 1.0

        elif (
            "halfcheetah" in env_entry
            or "walker2d" in env_entry
            or "hopper" in env_entry
        ):
            rewards = self.dataset.rewards

            episode_returns = self.get_returns()
            rewards /= np.max(episode_returns) - np.min(episode_returns)
            rewards *= 1000

            self.dataset.rewards = rewards

    def __len__(self):
        return len(self.dataset.observations)

    def __getitem__(self, idx):

        return Batch(
            states=self.dataset.observations[idx, :],
            next_states=self.dataset.next_observations[idx, :],
            actions=self.dataset.actions[idx, :],
            rewards=self.dataset.rewards[idx, :],
            masks=self.dataset.masks[idx, :],
        )
