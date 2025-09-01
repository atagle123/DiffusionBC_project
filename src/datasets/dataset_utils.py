import numpy as np
import collections
from dataclasses import dataclass, field
from typing import List, Dict
from src.datasets.normalization import GaussianNormalizer, MinMaxNormalizer

### D4RL utils ###

#  TODO ant maze v0 bugs with sequence dataset

def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['actions'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in ['observations', 'actions']: # for compatibility with v2 environments. only use observation and actions
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


### Trajectories dataset utils ###

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

@dataclass
class Episode:
    observations: List[float]
    actions: List[float]

@dataclass
class EpisodeDataset:
    episodes: List[Dict[str, Episode]] = field(default_factory=list)
    episodes_length: List[int] = field(default_factory=list)

    def add_episode(self, episode_data: Dict[str, Episode]): # TODO Episode class 
        """Add a new episode to the dataset."""
        episode_length = len(episode_data['observations'])
        
        for key in episode_data.keys():
            episode_data[key]= atleast_2d(episode_data[key])

        self.episodes_length.append(episode_length)
        self.episodes.append(episode_data)

    def get_episode(self, index: int) -> Dict[str, Episode]:
        """Retrieve an episode by index."""
        return self.episodes[index]

    def __len__(self):
        """Return the number of episodes in the dataset."""
        return len(self.episodes)
    
    def preprocess(self, history_len: int, pad_val: float = 0, normalization: str = "minmax", fields_to_normalize: list[str] = ["actions", "observations"]):
        self.normalize_fields(fields_to_normalize=fields_to_normalize, normalization=normalization) # normalize before padding
        self.shift_actions(pad_val=pad_val)
        self.pad(history_len=history_len, pad_val=pad_val)
    
    ### preprocessing methods ###

    def normalize_fields(self, fields_to_normalize: list[str] = ["actions", "observations"], normalization: str = "minmax"):
        """
        Normalize the specified fields in the dataset using the specified normalization method.

        Args:
            fields_to_normalize (list[str]): List of fields to normalize (e.g., "actions", "observations").
            normalization (str): Normalization method ("gaussian" or "minmax").
        """
        self.norm_params = self._get_normalization_params(fields_to_normalize=fields_to_normalize)
        self.normalizer = self._get_normalization_class(normalization)

        # Apply normalization to each episode
        for episode in self.episodes:
            for field in fields_to_normalize:
                episode[field] = self.normalizer.normalize(episode[field], field)
        
    def _get_normalization_params(self, fields_to_normalize):
        norm_params = {}

        # Compute normalization parameters for each field
        for field in fields_to_normalize:
            all_data = np.concatenate([ep[field] for ep in self.episodes], axis=0)
            params = {
                "mean": np.nanmean(all_data, axis=0),
                "std": np.nanstd(all_data, axis=0) + 1e-8,
                "max": np.max(all_data, axis=0),
                "min": np.min(all_data, axis=0),
            }
            norm_params[field] = params
        print(f"Normalization parameters: {norm_params}")
        return norm_params

    def _get_normalization_class(self, normalization):
        """
        Factory method to return the appropriate normalizer class.
        """
        if normalization == "gaussian":
            return GaussianNormalizer(self.norm_params)
        elif normalization == "minmax":
            return MinMaxNormalizer(self.norm_params)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    def shift_actions(self, pad_val: float = 0.0):
        """Shift actions to the right by one time step."""
        for episode in self.episodes:
            episode['actions'][1:,:] = episode['actions'][:-1,:]
            episode['actions'][0,:] = pad_val # zero padding at the beggining of actions    

    def pad(self, history_len: int, pad_val: int = 0, pad_fields: list[str] = ["actions", "observations"]):
        assert history_len>=1
        for episode_length, episode in zip(self.episodes_length, self.episodes): 
            for field in pad_fields:
                episode[field] = np.pad(episode[field], pad_width=((history_len-1, 0),(0,0)), constant_values=pad_val)
            
            episode_length += history_len - 1 # update lengths to account for padding