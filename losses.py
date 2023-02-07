import torch

from logger import get_logger
logger = get_logger("Losses logger")

class MSELoss(torch.nn.MSELoss):
    def __init__(self):
        super().__init__()