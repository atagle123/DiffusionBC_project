import os
import gym
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import d4rl.gym_mujoco
import d4rl.hand_manipulation_suite
import torch
from src.utils.arrays import batch_to_device
from src.agent.bc_agent import BC_Agent_Test
from src.agent.film_agent import FiLM_Agent
from src.utils.evaluation import evaluate_parallel
from src.datasets.bc_dataset import BC_Dataset
from src.datasets.trajectory_dataset import TrajectoriesDataset

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer:
    def __init__(self, cfg, logging_cfg):  
        self.cfg = cfg
        self.logging_cfg = logging_cfg 
        self.num_eval_episodes = 10
        self.val_dataset_ratio = 0.05

        if logging_cfg.wandb_log:
            wandb.init(
                project="Diffusion_BC",
                name=cfg.wandb.wandb_exp_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        self.dataset = TrajectoriesDataset(env_entry=cfg.dataset.env_entry)
       # self.dataset = BC_Dataset(env_entry=cfg.dataset.env_entry)
        action_dim = self.dataset.action_dim
        state_dim = self.dataset.observation_dim

        self.agent = FiLM_Agent( # TODO INSTANSIATE SEPARATELY... 
       # self.agent = BC_Agent_Test(
            action_dim=action_dim,
            state_dim=state_dim,
            cfg=cfg,
        )

        self.optimizer = torch.optim.AdamW(
            self.agent.diffusion_model.parameters(),
            lr=self.cfg.agent.training.train_lr,
            weight_decay=self.cfg.agent.training.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.agent.training.steps,
            eta_min=self.cfg.agent.training.eta_min
        )

        self.envs_for_eval = [
            gym.make(self.cfg.dataset.env_entry) for _ in range(self.num_eval_episodes)
        ]

        self._load_dataset() # TODO load dataset here

    def train(self):

        self._train_loop()
        wandb.finish()

    def _load_dataset(self):
        """
        Load the dataset and split it into training and validation subsets.
        """
        # Split dataset into training and validation subsets
        dataset_size = len(self.dataset)
        val_size = int(self.val_dataset_ratio * dataset_size) 
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.cfg.seed)
        )

        # Create dataloaders for training and validation
        self.train_dataloader = cycle(
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.agent.training.train_batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
        )

        self.val_dataloader = cycle(
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.cfg.agent.training.train_batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )
        )
         
    def _log_info(self, info: dict[str, float], step: int, prefix: str) -> None:
        info_str = " | ".join([f"{prefix}/{k}: {v:.4f}" for k, v in info.items()])
        print(f"{info_str} | (step {step})")

        if self.logging_cfg.wandb_log:
            wandb.log({f"{prefix}/{k}": v for k, v in info.items()}, step=step)

    def _train_loop(self):

        # Agent training
        for step in tqdm(range(1, self.cfg.agent.training.steps + 1), smoothing=0.1):
            self.optimizer.zero_grad(set_to_none=True)

            batch = next(self.dataloader)
            batch = batch_to_device(batch)
            loss, info = self.agent.diffusion_model.loss(*batch)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                if step % self.cfg.agent.training.update_ema_every == 0:
                    self.agent.step_ema(step)

                if step % self.logging_cfg.log_freq == 0:
                    self._log_info(info, step, prefix="train")

                if step % self.logging_cfg.save_freq == 0 or step == 1:
                    self.agent.save(step)

                if step % self.logging_cfg.eval_freq == 0 or step == 1: # log metrics in csv in the pc... or last step...
                    
                    self.agent.config_policy(batch_size = self.num_eval_episodes)  # configure the policy for evaluation
                    policy_fn = lambda state: self.agent.policy(state)

                    eval_info = evaluate_parallel(
                        policy_fn,
                        self.envs_for_eval,
                        self.cfg.dataset.env_entry,
                        num_episodes=self.num_eval_episodes,
                    )
                    self._log_info(eval_info, step, prefix="eval")

                