import cv2
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from configs.TrainConfig import TrainConfig
from transforms import SatTransforms
from utils import (
    make_train_step, 
    get_model, 
    get_dataset, 
    get_loss_fns, 
    get_tensorboard_writer, 
    check_cfg, 
    collate_fn,
    get_optimizer_scheduler
)

from logger import get_logger
logger = get_logger("Train logger")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn', force=True)
except RuntimeError:
    logger.warning("NOTE: UNKNOWN THING DIDN'T WORK.")

import pdb

def do_train(cfg, train_loader, loss_fns, model, optimizer, scheduler, writer):
    # Prepare to training
    iter_counter = 0
    det_loss_fn, count_loss_fn = loss_fns
    det_coeff = cfg.params['DET_COEFFICIENT']
    count_coeff = cfg.params['COUNT_COEFFICIENT']
    
    for epoch in range(cfg.params['N_EPOCHS']):
        training_bar = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
        for images_batch, anns_batch in training_bar:
            # Predict
            model.train()
            det_preds, n_objects_preds = model(images_batch)
            
#             # Save the predicted heatmap
#             if iter_counter % 300 == 0:
#                 # Plot the original heatmap over the original image
#                 img = (images_batch.squeeze().clone().detach().cpu().permute(1, 2, 0).numpy() * 255).astype(int)
#                 heatmap = det_preds.squeeze().clone().detach().cpu().numpy()
#                 fig, ax = plt.subplots()
#                 ax.imshow(img)
#                 heatmap_plot = ax.imshow(heatmap, cmap='coolwarm', alpha=0.5)
#                 cbar = plt.colorbar(heatmap_plot)
#                 fig.savefig(f"results/pred_map_overlayed_{iter_counter}.png")
                
#                 # Plot the thresholded heatmap over the original image
#                 tau = 0.95
#                 img = (images_batch.squeeze().clone().detach().cpu().permute(1, 2, 0).numpy() * 255).astype(int)
#                 heatmap = (det_preds * (det_preds > 0.8).float()).squeeze().clone().detach().cpu().numpy()
#                 fig, ax = plt.subplots()
#                 ax.imshow(img)
#                 heatmap_plot = ax.imshow(heatmap, cmap='coolwarm', alpha=0.5)
#                 cbar = plt.colorbar(heatmap_plot)
#                 fig.savefig(f"results/pred_map_overlayed_thresholded_{iter_counter}.png")
                
#                 save_image(det_preds, f"results/pred_map_{iter_counter}.png")
#                 logger.debug(f"Number of predicted objects: {n_objects_preds.item()}")
            
            # Prepare to compute the losses
            orig_sizes = torch.tensor(
                [[cfg.params['IMG_HEIGHT'], cfg.params['IMG_WIDTH']] for _ in range(images_batch.shape[0])],
                device=device
            )
            n_gt_pts = torch.tensor(
                [len(sample) for sample in anns_batch],
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)
            
            # Compute the loss
            det_loss = det_loss_fn(det_preds, anns_batch)
            count_loss = count_loss_fn(n_objects_preds, n_gt_pts)
            
            # Apply the weights
            det_loss_weighted = det_coeff * det_loss
            count_loss_weighted = count_coeff * count_loss
            
            # Add up the loss terms
            total_loss = det_loss_weighted + count_loss_weighted
            
            # Backpropagate the loss
            total_loss.backward()
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad()
            
            # Log the info
            writer.add_scalar("Detection loss", det_loss_weighted, iter_counter)
            writer.add_scalar("Counter loss", count_loss_weighted, iter_counter)
            
            iter_counter += 1

if __name__ == "__main__":
    """Set up the configurations"""
    cfg = TrainConfig()
    check_cfg(cfg, verbose=cfg.params['VERBOSE'])
    
    """Load the dataset"""
    satellite_transforms = SatTransforms(cfg.dataset_params['APPLY_TRAIN_TRANSFORMS'])
    train_transform = satellite_transforms.get_train_transforms()
    train_set = get_dataset(cfg, transform=train_transform, debug_on=cfg.params['DEBUG'], device=device)
    
    """Initialize the dataloaders"""
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.params['BATCH_SIZE'], 
        shuffle=cfg.params['SHUFFLE'],
        collate_fn=collate_fn
    )
    
    """Initialize the model"""
    model = get_model(cfg, device=device)
    
    """Optimizer, loss function, evaluator"""
    loss_fns = get_loss_fns(cfg, device=device) # (det_loss, class_loss)
    optimizer, scheduler = get_optimizer_scheduler(
        model, cfg.params['BASE_LR'], cfg.params['LR_GAMMA'], cfg.params['LR_SCHEDULING_ON']
    )
    
    """Training"""
    writer = get_tensorboard_writer(cfg.logging_params['LOG_DIR'])
    do_train(cfg, train_loader, loss_fns, model, optimizer, scheduler, writer)
    
    """Close the tensorboard logger"""
    writer.flush()
    writer.close()