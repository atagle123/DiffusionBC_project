from omegaconf import DictConfig
from src.agent.agent import Agent
import torch
from src.utils.arrays import DEVICE
import os
import copy
from src.models.inpainting_networks import ScoreModel_test, TemporalUnet, DiT
from src.models.inpainting_diffusion import GaussianInpaintingDiffusion
from src.utils.ema import EMA
from src.utils.arrays import report_parameters
from .film_agent import FiLM_Agent, HistoryBuffer
from src.datasets.normalization import BaseNormalizer
import numpy as np


class InpaintHistoryBuffer(HistoryBuffer):
    def __init__(
        self,
        normalizer: BaseNormalizer,
        history_len: int,
        action_dim: int,
        obs_dim: int,
        batch_size: int,
        pad_value: float = 0.0,
        device: str = DEVICE,
    ):
        super().__init__(
            normalizer, history_len, action_dim, obs_dim, batch_size, pad_value, device
        )

        self.curr_len = 0

    def add_state(self, observations: np.ndarray):  # needs to handles batch size>1
        super().add_state(observations)
        self.curr_len = min(self.curr_len + 1, self.history_len)

    def __call__(self) -> np.ndarray:
        return self.history[:, -self.curr_len :, :]

    @property
    def len(self) -> int:
        return self.curr_len


class Inpainting_Agent(FiLM_Agent):
    def __init__(self, action_dim: int, state_dim: int, cfg: DictConfig):
        os.makedirs(cfg.agent.savepath, exist_ok=True)

        self.cfg = cfg
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.history_len = cfg.dataset_configs.history_len
        self.horizon = cfg.dataset_configs.horizon

        model = globals()[cfg.method.agent_architecture](
            horizon=self.horizon,
            transition_dim=action_dim + state_dim,
            **cfg.diffusion_network,
        ).to(DEVICE)  # NOTE it is neccesary to sendto device?

        report_parameters(model)

        self.diffusion_model = GaussianInpaintingDiffusion(
            model=model, data_dim=action_dim + state_dim, **cfg.agent.diffusion
        ).to(DEVICE)

        self.ema = EMA(self.cfg.agent.training.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)

    def _init_history_buffer(self, batch_size: int):
        """
        Initializes a history buffer with the current state and action.

        """
        self.history_buffer = InpaintHistoryBuffer(
            normalizer=self.normalizer,
            history_len=self.history_len,
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

        actions = (
            samples[:, self.history_buffer.len, : self.action_dim].cpu().numpy()
        )  # action that follows curr history
        self.history_buffer.add_action(actions)
        actions = self.normalizer.unnormalize(actions, "actions")

        return actions  # Return the first action
