from src.datasets.dataset_utils import sequence_dataset, EpisodeDataset
import numpy as np
import gym
import torch
from collections import namedtuple
from typing import List

Batch = namedtuple("Batch", "trajectory mask")


class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, env_entry, horizon=16, stride=1, pad_val=0, mask_cache_size=100000,
        p_mask_set=[0.15,0.35,0.55,0.75,0.95], mask_strategy='all', unmask_start=False, mask_end=False, **kwargs):

        self.horizon = horizon

        self.env = gym.make(env_entry)
        dataset_itr = sequence_dataset(self.env)

        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        
        dataset = EpisodeDataset()
        for _, episode in enumerate(dataset_itr):
            dataset.add_episode(episode_data = episode) # TODO discard episodes lesser than horizon

        # TODO manage max episode length... and max n episodes
        dataset.preprocess(history_len=0, pad_val=pad_val)

        self.normalizer = dataset.normalizer # Get normalizer from dataset after preprocessing, to use in evaluation

        lengths_list = dataset.episodes_length
        self.indices = self.make_indices(lengths_list, traj_len=horizon, stride=stride)

        self.dataset = dataset
        self.get_env_attributes()

        self.p_mask_set = p_mask_set
        self.mask_strategy = mask_strategy
        self.unmask_start = unmask_start
        self.mask_end = mask_end
        self.mask_cache_size = mask_cache_size
        self.make_masks()
    
    def make_masks(self):
        # Generate a cache of random masks: False where an element is missing
        self.mask_idx = 0

        # Random masking: False where x is missing
        n_masked = np.random.choice(self.p_mask_set, self.mask_cache_size) * self.horizon
        n_masked = n_masked.astype(np.int_)

        if self.mask_strategy == 'all':
            self.masks = np.zeros((self.horizon, self.action_dim+self.state_dim), dtype=bool)
        elif self.mask_strategy == 'half':
            self.masks = np.zeros((self.horizon, self.action_dim+self.state_dim), dtype=bool)
            self.masks[:self.horizon//2,:] = True
        elif self.mask_strategy == 'random_split':  # random split between history and future
            self.masks = np.zeros((self.mask_cache_size, self.horizon, self.action_dim+self.state_dim), dtype=bool)
            n_masked = np.random.randint(self.horizon, size=self.mask_cache_size) # allow possibility of masking everything
            for i in range(self.mask_cache_size):
                self.masks[i,:n_masked[i],:] = True
        elif self.mask_strategy == 'joint':  # all feature dims jointly masked
            self.masks = np.ones((self.mask_cache_size, self.horizon, self.action_dim+self.state_dim), dtype=bool)
            for i in range(self.mask_cache_size):
                idx = np.random.permutation(self.horizon)[:n_masked[i]]
                self.masks[i,idx,:] = False
        elif self.mask_strategy == 'indep_state_action': # masking independent at level of state and action only
            mask_a = np.ones((self.mask_cache_size, self.horizon, self.action_dim), dtype=bool)
            mask_s = np.ones((self.mask_cache_size, self.horizon, self.state_dim), dtype=bool)

            for i in range(self.mask_cache_size):
                idx = np.random.permutation(self.horizon)[:n_masked[i]]
                mask_a[i,idx,:] = False

                idx = np.random.permutation(self.horizon)[:n_masked[i]]
                mask_s[i,idx,:] = False

            self.masks = np.concatenate([mask_a, mask_s], axis=-1)
        elif self.mask_strategy == 'random':  # each feature dim independently masked
            self.masks = np.ones((self.mask_cache_size, self.horizon, self.action_dim+self.state_dim), dtype=bool)
            for i in range(self.mask_cache_size):
                n = n_masked[i]
                for j in range(self.action_dim+self.observation_dim):
                    idx = np.random.permutation(self.horizon)[:n]
                    self.masks[i,idx,j] = False
        else:
            raise ValueError(f'Unrecognised masking strategy {self.mask_strategy}')

        if self.mask_end and self.mask_strategy not in ['all', 'half' 'random_split']:
            self.masks[:,-1,:] = False
        
        if self.unmask_start:
            if self.mask_strategy == 'all':
                self.masks[0,:] = True
            else:
                self.masks[:,0,:] = True

    def make_indices(self, episodes_lengths: list, traj_len: int, max_traj_len: int = 1000, stride: int = 1):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
            assume that the dataset is already padded
            traj_len = horizon
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

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.mask_strategy in ['all', 'half']:
            mask = self.masks
        else:
            mask = self.masks[self.mask_idx]
            self.mask_idx += 1
            if self.mask_idx == self.mask_cache_size:
                self.make_masks()

        return Batch(trajectories, mask)

