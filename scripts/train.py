import logging
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig

from src.utils.training import Trainer


# ---------------------------- #
#     Constants & Defaults     #
# ---------------------------- #


@dataclass(frozen=True)
class Defaults:
    config_path: str = "../configs/D4RL"
    wandb_log: bool = False
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
    wandb_log: bool
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
        wandb_log=defaults.wandb_log,
        val_dataset=defaults.val_dataset
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

    trainer = Trainer(cfg, logging_config)
    trainer.train()

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
