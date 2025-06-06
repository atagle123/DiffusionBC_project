from src.agent.agent import Agent
import torch
from src.utils.arrays import DEVICE
import os
import copy
from src.models.temporal import ScoreModel_test
from src.models.diffusion import GaussianDiffusion
from src.utils.ema import EMA
from src.utils.arrays import report_parameters


class BC_Agent_Test(Agent):
    def __init__(self, action_dim, state_dim, cfg):

        os.makedirs(cfg.agent.savepath, exist_ok=True)

        self.cfg = cfg
        model = ScoreModel_test(
            data_dim=action_dim,
            state_dim=state_dim,
            **cfg.agent.diffusion_network
        ).to(
            DEVICE
        )  # NOTE it is neccesary to sendto device?

        report_parameters(model)

        self.diffusion_model = GaussianDiffusion(
            model=model,
            data_dim=action_dim,
            **cfg.agent.diffusion
        ).to(DEVICE)

        self.ema = EMA(self.cfg.agent.training.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.diffusion_model.state_dict())

    def step_ema(self, step):
        if step < self.cfg.agent.training.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_model)

    def save(self, step):
        """
        saves model and ema to disk;
        """
        data = {
            "model": self.diffusion_model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.cfg.agent.savepath, f"state_{step}.pt")
        torch.save(data, savepath)

        print(f"Saved model to {savepath}", flush=True)

    def load(self, step):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.actor_savepath, f"state_{step}.pt")
        data = torch.load(loadpath)

        self.diffusion_model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def load_latest_step(self, step):
        if step == "latest":
            step = self.get_latest_step(self.actor_savepath)
        self.load(step)

    def get_latest_step(self, loadpath):
        import glob

        states = glob.glob1(*loadpath, "state_*")
        latest_step = -1
        for state in states:
            step = int(state.replace("state_", "").replace(".pt", ""))
            latest_step = max(step, latest_step)
        return latest_step

    def config_policy(self):
        # assumes that the savepath has trainer, model, diffusion and dataset configs

        self.diffusion_model.setup_sampling()

    def policy(self, state):
        """
        Generates an action based on the provided state using a diffusion model.

        Args:
        state (array-like): The current state from the environment that needs to be procsessed.

        Returns:
        numpy.ndarray: Generated Action
        """

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        samples = self.diffusion_model(state=state).sample.detach()

        actions = samples
        return actions.cpu().numpy()  # Return the first action
