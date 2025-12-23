"""
Exponential Moving Average for PyTorch models.
"""


class EMA:
    """
    Exponential Moving Average for model parameters.

    EMA maintains a shadow copy of model parameters that is updated with:
        shadow = beta * shadow + (1 - beta) * current_param

    This provides more stable predictions and better generalization.
    Higher beta (e.g., 0.999) means slower updates and more stability.
    """

    def __init__(self, model, beta=0.999):
        self.model = model
        self.beta = beta
        self.shadow = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

    def update(self, model):
        """Update shadow parameters with current model parameters"""
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.beta).add_(param.data, alpha=1 - self.beta)

    def copy_to(self, model):
        """Copy shadow parameters to model (for inference)"""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
