import logging
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig

from src.utils.training import Trainer
from src.datasets import BC_Dataset, TrajectoriesDataset, MaskedDataset
from src.agent import BC_Agent_Test, Inpainting_Agent, FiLM_Agent
import torch

# ---------------------------- #
#     Constants & Defaults     #
# ---------------------------- #


@dataclass(frozen=True)
class Defaults:
    config_path: str = "../configs/D4RL"
    val_dataset: bool = True
    log_freq: int = 10000
    save_freq: int = 10000
    eval_freq: int = 10000


DEFAULTS = Defaults()


# ---------------------------- #
#       Training Config        #
# ---------------------------- #


@dataclass
class TrainingConfig:
    log_freq: int
    save_freq: int
    eval_freq: int
    val_dataset: bool


# ---------------------------- #
#     Utility Functions        #
# ---------------------------- #


def build_training_config(defaults: Defaults) -> TrainingConfig:
    """
    Build a training config object from the defaults.
    """
    return TrainingConfig(
        log_freq=defaults.log_freq,
        save_freq=defaults.save_freq,
        eval_freq=defaults.eval_freq,
        val_dataset=defaults.val_dataset,
    )


def build_trainer(cfg: DictConfig, logging_config):
    dataset = globals()[cfg.method.dataset_class](
        env_entry=cfg.dataset.env_entry, **cfg.dataset_configs
    )
    action_dim = dataset.action_dim
    state_dim = dataset.observation_dim

    agent = globals()[cfg.method.agent_class](
        action_dim=action_dim,
        state_dim=state_dim,
        cfg=cfg,
    )

    optimizer = torch.optim.AdamW(
        agent.diffusion_model.parameters(),
        lr=cfg.agent.training.train_lr,
        weight_decay=cfg.agent.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.agent.training.steps, eta_min=cfg.agent.training.eta_min
    )

    return Trainer(
        cfg,
        logging_config,
        dataset=dataset,
        agent=agent,
        optimizer=optimizer,
        scheduler=scheduler,
    )


# ---------------------------- #
#        Main Script           #
# ---------------------------- #

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path=DEFAULTS.config_path, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training execution.
    """
    logger.info("Starting experiment with configuration: %s", cfg)

    logging_config = build_training_config(DEFAULTS)

    trainer = build_trainer(cfg, logging_config)
    trainer.train()

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
