from omegaconf import DictConfig
from src.models.inpainting_networks import TemporalUnet
from src.agent.agent import Agent
import torch
from src.utils.arrays import DEVICE
import os
import copy
from src.models.networks import ScoreModel_test
from src.models.diffusion import GaussianDiffusion
from src.utils.ema import EMA
from src.utils.arrays import report_parameters
import numpy as np
from src.datasets.normalization import BaseNormalizer
from src.utils.history_buffer import HistoryBuffer


class BC_Agent_Test(Agent):
    def __init__(self, action_dim: int, state_dim: int, cfg: DictConfig):
        os.makedirs(cfg.agent.savepath, exist_ok=True)

        self.cfg = cfg
        model = ScoreModel_test(
            data_dim=action_dim, state_dim=state_dim, **cfg.diffusion_network
        ).to(DEVICE)  # NOTE it is neccesary to sendto device?

        report_parameters(model)

        self.diffusion_model = GaussianDiffusion(
            model=model, data_dim=action_dim, **cfg.agent.diffusion
        ).to(DEVICE)

        self.ema = EMA(self.cfg.agent.training.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)

    def config_policy(self, batch_size: int, normalizer: BaseNormalizer):
        # assumes that the savepath has trainer, model, diffusion and dataset configs

        self.diffusion_model.setup_sampling()

    def policy(self, state: np.ndarray) -> np.ndarray:
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


class BCTrajectoryAgent(Agent):
    def __init__(self, action_dim: int, state_dim: int, cfg: DictConfig):
        os.makedirs(cfg.agent.savepath, exist_ok=True)

        self.cfg = cfg
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = cfg.dataset_configs.horizon

        model = TemporalUnet(
            horizon=self.horizon,
            transition_dim=action_dim + state_dim,
            **cfg.diffusion_network,
        ).to(DEVICE)  # NOTE it is neccesary to sendto device?

        report_parameters(model)

        self.diffusion_model = GaussianDiffusion(
            model=model, data_dim=action_dim + state_dim, **cfg.agent.diffusion
        ).to(DEVICE)

        self.ema = EMA(self.cfg.agent.training.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)

    def config_policy(self, batch_size: int, normalizer: BaseNormalizer):
        self.diffusion_model.setup_sampling(horizon=self.horizon)
        self.normalizer = normalizer
        self._init_history_buffer(batch_size)

    def _init_history_buffer(self, batch_size: int):
        """
        Initializes a history buffer with the current state and action.

        """
        self.history_buffer = HistoryBuffer(
            normalizer=self.normalizer,
            history_len=0,
            action_dim=self.action_dim,
            obs_dim=self.state_dim,
            batch_size=batch_size,
        )

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Generates an action based on the provided state using a diffusion model.

        Args:
        state (array-like): The current state from the environment that needs to be procsessed.

        Returns:
        numpy.ndarray: Generated Action
        """

        self.history_buffer.add_state(state)
        samples = self.diffusion_model(condition=self.history_buffer()).sample.detach()

        actions = samples[:, 0, : self.action_dim].cpu().numpy()  # first action
        self.history_buffer.add_action(actions)
        actions = self.normalizer.unnormalize(actions, "actions")
        return actions  # Return the first action
