import torch
from torchsummary import summary

from configs.TrainConfig import TrainConfig
from transforms import SatTransforms
from utils import make_train_step, get_model, get_dataset, get_loss_fn

from logger import get_logger
logger = get_logger("Train logger")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

if __name__ == "__main__":
    """Set up the configurations"""
    cfg = TrainConfig()
    
    """Load the dataset"""
    satellite_transforms = SatTransforms(cfg.dataset_params['APPLY_TRAIN_TRANSFORMS'])
    train_transform = satellite_transforms.get_train_transforms()
    train_set = get_dataset(cfg, transform=train_transform, device=device)
    
    """Initialize the dataloaders"""
    pass
    
    """Initialize the model"""
    model = get_model(cfg)
    
    """Optimizer, loss function, evaluator"""
    loss_fn = get_loss_fn(cfg.params['LOSS_FN'])
    # train_step = make_train_step()
    
    """Training"""
    if cfg.params['EVAL_ONLY']:
        pass
    
    """Evaluate"""
    pass