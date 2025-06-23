from src.agent.agent import Agent
import torch
from src.utils.arrays import DEVICE
import os
import copy
from src.models.networks import FiLMTemporalUnet
from src.models.diffusion import FiLMGaussianDiffusion
from src.utils.ema import EMA
from src.utils.arrays import report_parameters


class FiLM_Agent(Agent):
    def __init__(self, action_dim, state_dim, cfg):

        os.makedirs(cfg.agent.savepath, exist_ok=True)

        self.cfg = cfg
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
            data_dim=action_dim,
            **cfg.agent.diffusion
        ).to(DEVICE)

        self.ema = EMA(self.cfg.agent.training.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)

    def config_policy(self):

        self.diffusion_model.setup_sampling() # TODO 

    def policy(self, state): # TODO... 
        """
        Generates an action based on the provided state using a diffusion model.

        Args:
        state (array-like): The current state from the environment that needs to be procsessed.

        Returns:
        numpy.ndarray: Generated Action
        """

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        samples = self.diffusion_model(condition=state).sample.detach()

        actions = samples
        return actions.cpu().numpy()  # Return the first action


        # TODO add history buffer to agent... 