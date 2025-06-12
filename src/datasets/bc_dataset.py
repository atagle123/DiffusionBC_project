from src.datasets.d4rl_dataset import D4RL_Dataset
from collections import namedtuple

BC_batch = namedtuple("BC_batch", "action state")


class BC_Dataset(D4RL_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.dataset.observations)

    def __getitem__(self, idx):

        return BC_batch(
            action=self.dataset.actions[idx, :], state=self.dataset.observations[idx, :]
        )