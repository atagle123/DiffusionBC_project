from abc import ABC, abstractmethod
import os
import torch

class Agent(ABC):
    def __init__(self):
        pass

    ## ema methods ##
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.diffusion_model.state_dict())

    def step_ema(self, step):
        if step < self.cfg.agent.training.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_model)

    ### save/load methods ###

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

    ### inference ###
    @abstractmethod
    def config_policy(self):
        pass

    @abstractmethod
    def policy(self):
        pass
