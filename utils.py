from models_zoo.PointDetector.unet_model import UNet
from losses import MSELoss
from dataset import SatelliteDataset

from logger import get_logger
logger = get_logger("Utils logger")

import pdb

def make_train_step():
    pass

def get_loss_fn(loss_fn_keyword):
    if loss_fn_keyword == "MSELoss":
        loss_fn = MSELoss()
    else:
        raise NotImplementedError
    logger.info(f"Using {loss_fn_keyword} loss.")
    return loss_fn

def get_model(cfg):
    if cfg.params["MODEL_NAME"] == "UNet":
        model = UNet(
            n_channels=cfg.params['N_CHANNELS'],
            n_classes=cfg.params['N_CLASSES'],
            height=cfg.params['IMG_HEIGHT'],
            width=cfg.params['IMG_WIDTH']
        )
    else:
        raise NotImplementedError
    logger.info(f"Loaded {cfg.params['MODEL_NAME']} model.")
    return model

def get_dataset(cfg, transform=None, device='cpu'):
    dataset = SatelliteDataset(
        cfg.dataset_params['DATASET_DIR'],
        transform=transform,
        device=device
    )
    logger.info(f"Loaded dataset.")
    return dataset