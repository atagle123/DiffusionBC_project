import numpy as np
import d4rl
import gym
import torch
from dataclasses import dataclass
from collections import namedtuple
from src.datasets.dataset_utils import sequence_dataset
from dataclasses import dataclass, field
from typing import List, Dict

Batch = namedtuple('Batch', 'trajectories conditions')


def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


@dataclass
class Episode:
    observations: List[float]
    actions: List[float]
    rewards: List[float]

@dataclass
class EpisodeDataset:
    episodes: List[Dict[str, Episode]] = field(default_factory=list)
    episodes_lenght: List[int] = field(default_factory=list)

    def add_episode(self, episode_data: Dict[str, Episode]): # TODO CHANGE CLASSSES MANAGEMENT
        """Add a new episode to the dataset."""
        episode_length = len(episode_data['observations'])
        
        for key in episode_data.keys():
            episode_data[key]= atleast_2d(episode_data[key])

        self.episodes_lenght.append(episode_length)
        self.episodes.append(episode_data)

    def get_episode(self, index: int) -> Dict[str, Episode]:
        """Retrieve an episode by index."""
        return self.episodes[index]

    def __len__(self):
        """Return the number of episodes in the dataset."""
        return len(self.episodes)
    
    ### normalization

    def normalize(self):
        
        for episode in self.episodes:
            pass



class TrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(self, env_entry, clip_actions_to_eps=True, history_len=1, horizon=16, stride=1,
        max_n_episodes=10000, pad_val=0):

        self.history_len = history_len
        self.horizon = horizon


        self.env = gym.make(env_entry)
        dataset_itr = sequence_dataset(self.env) # TODO pass to atleast 2d...


        dataset = EpisodeDataset()
        for i, episode in enumerate(dataset_itr):
            dataset.add_episode(episode_data = episode) # Add all episodes to a Dataset and apply normalization of rewards and of actions... also atleast 2d
        # normalization... 
        # pad later than normalization
        # pad short episodes to have at least history at left.
        # pad left and right or only left? 

        # TODO manage cases when episodes has least lenght than horizon or history_len or the sum of it
        # TODO manage max episode lenght... and max n episodes

        lenghts_list = dataset.episodes_lenght
        self.indices = self.make_indices(lenghts_list, traj_len=history_len+horizon, stride=stride)

        self.dataset = dataset
        self.get_env_attributes() # TODO resuse from base class

    def make_indices(self, episodes_lengths: list, traj_len: int, max_traj_len: int = 1000, stride: int = 1):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
            assume that the dataset is already padded
            traj_len = history_len+horizon
        '''
        indices = []
        for i, path_length in enumerate(episodes_lengths):

            max_start = min(path_length, max_traj_len)-traj_len
            assert max_start>=0 # TODO pad min...
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

    def __len__(self):
        return(len(self.indices))

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.dataset.get_episode(path_ind)["observations"][start:end] # TODO maybe join both arrays for faster indexing... 
        actions = self.dataset.get_episode(path_ind)["actions"][start:end]

        full_array = np.concatenate([actions, observations], axis=-1)
        conditions = full_array[:self.history_len, :]
        trajectories = full_array[self.history_len:, :]

        return Batch(trajectories, conditions)