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
    def __init__(self, normalizer, history_len, action_dim, obs_dim, batch_size: int = 10):
        self.normalizer = normalizer # TODO
        self.history_len = history_len
        self.action_dim = action_dim
        self.latest_known_index=history_len # or 0 
#        initial_observation=self.normalizer.normalize(initial_observation, 'observations')

        self.history = np.zeros((batch_size, history_len, obs_dim + action_dim), dtype=np.float32) # zero padding

        #self.history[:,-1, -obs_dim:] = initial_observation # shift when add action
    
    def add_state(self, observations): # needs to handles batch size>1
        observations = self._expand_dims(observations) # add at least 2d check dims B, 1, O
        #observations = self.normalizer.normalize(observations, 'observations')

        self.history[:,-1, self.action_dim:] = observations  # TODO test this

    def add_action(self, actions):
        actions = self._expand_dims(actions) # no normalize when adding actions, shift when adding actions

        self.history[:, :-1] = self.history[:, 1:]  # shift CHECK 

        self.history[:,-1, :self.action_dim] = actions # add actions TODO test this

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
        model = FiLMTemporalUnet(
            #data_dim=action_dim,
            #state_dim=state_dim,
            horizon = 16,
            history_len = 4, #TODO history len
            transition_dim=action_dim+state_dim,
            cond_dim = state_dim, # cond dim is not used... 
            #**cfg.agent.diffusion_network
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

    def config_policy(self):

        self.diffusion_model.setup_sampling() # TODO
        self._init_history_buffer()
    
    def _init_history_buffer(self):
        """
        Initializes a history buffer with the current state and action.

        """
        self.history_buffer =  HistoryBuffer(
            normalizer= None, # TODO 
            history_len=4, #self.cfg.agent.diffusion.history_len,
            action_dim=self.action_dim,
            obs_dim=self.state_dim,
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
        return actions  # Return the first action # unnormalize