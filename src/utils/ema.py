import torch.nn as nn


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module):
        for current_params, ema_params in zip(
            current_model.parameters(), ema_model.parameters()
        ):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
