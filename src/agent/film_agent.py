from src.agent.agent import Agent
import torch
from src.utils.arrays import DEVICE
import os
import copy
from src.models.networks import FiLMTemporalUnet
from src.models.diffusion import FiLMGaussianDiffusion
from src.utils.ema import EMA
from src.utils.arrays import report_parameters
import numpy as np

class HistoryBuffer: 
    def __init__(self, normalizer, history_len: int, action_dim: int, obs_dim: int, batch_size: int, pad_value: float = 0.0):
        self.normalizer = normalizer
        self.history_len = history_len
        self.action_dim = action_dim
        self.latest_known_index=history_len # or 0 

        self.history = np.full((batch_size, history_len, obs_dim + action_dim), pad_value, dtype=np.float32)  # fill with pad value

    
    def add_state(self, observations): # needs to handles batch size>1
        observations = self._expand_dims(observations) # add at least 2d check dims B, 1, O
        observations = self.normalizer.normalize(observations, 'observations')

        self.history[:,-1, self.action_dim:] = observations

    def add_action(self, actions):
        """
        Adds actions to the history buffer and shifts the buffer to the left to make room for new actions.

        The history buffer maintains a sequence of past actions and observations. When a new action is added:
        - The entire buffer is shifted left by one position, discarding the oldest entry.
        - The new action is inserted at the last position in the buffer.
        - Actions are not normalized here, as normalization is handled elsewhere.
        - The typical usage pattern is: add_state -> add_action (with shift) -> repeat.
        """
        actions = self._expand_dims(actions)

        self.history[:, :-1] = self.history[:, 1:] # shift history to the left

        self.history[:,-1, :self.action_dim] = actions

    def _expand_dims(self, arr, min_dims=2):
        while arr.ndim < min_dims:
            arr = np.expand_dims(arr, axis=0)  # add dim at the front
        return arr

    def __call__(self, device = "cuda:0"):
        return torch.tensor(self.history, dtype=torch.float32, device=device)

class FiLM_Agent(Agent):
    def __init__(self, action_dim, state_dim, cfg):

        os.makedirs(cfg.agent.savepath, exist_ok=True)

        self.cfg = cfg
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.history_len = cfg.dataset_configs.history_len
        self.horizon = cfg.dataset_configs.horizon

        model = FiLMTemporalUnet(
            horizon = self.horizon,
            history_len = self.history_len,
            transition_dim=action_dim+state_dim,
            cond_dim = state_dim, # cond dim is not used... 
            **cfg.diffusion_network
        ).to(
            DEVICE
        )  # NOTE it is neccesary to sendto device?

        report_parameters(model)

        self.diffusion_model = FiLMGaussianDiffusion(
            model=model,
            data_dim=action_dim+state_dim,
            **cfg.agent.diffusion
        ).to(DEVICE)

        self.ema = EMA(self.cfg.agent.training.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)

    def config_policy(self, batch_size:int, normalizer):

        self.diffusion_model.setup_sampling()
        self.normalizer = normalizer
        self._init_history_buffer(batch_size)
    
    def _init_history_buffer(self, batch_size: int):
        """
        Initializes a history buffer with the current state and action.

        """
        self.history_buffer =  HistoryBuffer(
            normalizer= self.normalizer,
            history_len=self.history_len,
            action_dim=self.action_dim,
            obs_dim=self.state_dim,
            batch_size=batch_size, 
        )

    def policy(self, state): # TODO... 
        """
        Generates an action based on the provided state using a diffusion model.

        Args:
        state (array-like): The current state from the environment that needs to be procsessed.

        Returns:
        numpy.ndarray: Generated Action
        """

        self.history_buffer.add_state(state)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        samples = self.diffusion_model(condition=self.history_buffer()).sample.detach()

        actions = samples[:,0, :self.action_dim].cpu().numpy() # first action
        self.history_buffer.add_action(actions)
        actions = self.normalizer.unnormalize(actions, 'actions')
        return actions  # Return the first action