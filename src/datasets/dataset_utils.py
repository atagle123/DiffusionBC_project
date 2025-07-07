import numpy as np
import collections
from dataclasses import dataclass, field
from typing import List, Dict

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
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
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
