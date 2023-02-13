import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models_zoo.PointDetector.unet_model import UNet
from losses import MSELoss, WeightedHausdorffDistance, SmoothL1Loss
from dataset import SatelliteDataset

from logger import get_logger
logger = get_logger("Utils logger")

import pdb

def collate_fn(samples):
    images = []
    anns = []
    
    for sample in samples:
        # Extract the sample image and annotations
        image = sample[0]
        ann = sample[1]
        
        # Add to the lists of images and annotations
        images.append(image)
        anns.append(ann)
    
    images = torch.stack(images)
    
    return (images, anns)

def check_cfg(cfg, verbose=0):
    # Create the required directories
    required_dirs = []
    required_dirs.append(cfg.params['OUTPUT_DIR'])
    required_dirs.append(cfg.logging_params['LOG_DIR'])
    
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        if verbose > 0:
            logger.debug(f"Created {directory}.")

def get_loss_fns(cfg, device='cpu'):
    def get_loss_fn(loss_fn_keyword, det_vs_count):
        if loss_fn_keyword == "MSELoss":
            loss_fn = MSELoss(device=device)
        elif loss_fn_keyword == "WeightedHausdorffDistance":
            loss_fn = WeightedHausdorffDistance(
                resized_height=cfg.params['IMG_HEIGHT'], 
                resized_width=cfg.params['IMG_WIDTH'],
                device=device
            )
        elif loss_fn_keyword == "SmoothL1Loss":
            loss_fn = SmoothL1Loss()
        else:
            raise NotImplementedError
        logger.info(f"Using {loss_fn_keyword} loss for {det_vs_count}.")
        return loss_fn
    
    # Extract the loss names
    det_loss = cfg.params['DET_LOSS_FN']
    count_loss = cfg.params['COUNT_LOSS_FN']
    
    # Generate the loss function
    det_loss_fn = get_loss_fn(det_loss, "detection")
    count_loss_fn = get_loss_fn(count_loss, "object count")
    
    return (det_loss_fn, count_loss_fn)

def get_model(cfg, device='cpu'):
    if cfg.params["MODEL_NAME"] == "UNet":
        model = UNet(
            n_channels=cfg.params['N_CHANNELS'],
            n_classes=cfg.params['N_CLASSES'],
            height=cfg.params['IMG_HEIGHT'],
            width=cfg.params['IMG_WIDTH']
        )
        model = model.to(device)
    else:
        raise NotImplementedError
    logger.info(f"Loaded {cfg.params['MODEL_NAME']} model.")
    return model

def get_dataset(cfg, transform=None, debug_on=False, device='cpu'):
    dataset = SatelliteDataset(
        cfg.dataset_params['DATASET_DIR'],
        transform=transform,
        debug_on=debug_on,
        device=device
    )
    logger.info(f"Loaded a dataset with {len(dataset)} images.")
    return dataset

def get_dataloader(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0, shuffle=False):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        drop_last=False, 
        shuffle=False, 
        sampler=sampler,
        collate_fn=collate_fn
    )
    return dataloader

def get_tensorboard_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Initialized a TensorBoard writer with the logging directory at {log_dir}.")
    return writer

def get_optimizer_scheduler(model, base_lr, lr_gamma, scheduled=False):
    if scheduled:
        raise NotImplementedError
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = None
        return (optimizer, scheduler)