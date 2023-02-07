import os
import glob
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

from torch.utils.data import Dataset

from logger import get_logger
logger = get_logger("Dataset logger")

import pdb

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, transform=None, shuffle=True, device='cpu'):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform
        self.metadata = self.extract_metadata(data_dir, shuffle=shuffle)
        self.device = device
        
    def extract_metadata(self, data_dir, shuffle=True):
        """
        This method extract metadata which described the detection dataset.
        
        Expecting the dataset to have the following structure:
        └── data_dir
            ├── annotations
            ├── images
        Each image in the "images" directory has an analogous annotation file in the "annotations" directory.
        For example: image_1234.jpg <-> image_1234.pkl.
        """
        metadata = []
        images_dir = os.path.join(data_dir, "images")
        annotations_dir = os.path.join(data_dir, "annotations")
        
        # Load the images and annotations list
        images_list = []
        formats_list = ['jpg', 'png']
        for image_format in formats_list:
            images_list += glob.glob(images_dir + f"/*.{image_format}")
        annotations_list = glob.glob(annotations_dir + "/*.pkl")
        if len(images_list) != len(annotations_list): logger.error("Different number of images and annotation files!")
        
        # Load the metadata
        logger.info("Loading the dataset...")
        for image_path in tqdm(images_list):
            # Get the annotation file path
            file_code = image_path.split('/')[-1].split('.')[0]
            annotation_path = os.path.join(annotations_dir, f"{file_code}.pkl")
            
            with open(annotation_path, 'rb') as f:
                annotations = pickle.load(f)
                
            # Leave only small vehicles
            annotations = annotations['small']
            
            metadata_ = {
                "image_path": image_path,
                "annotations": annotations
            }
            
            metadata.append(metadata_)
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        data = self.metadata[idx]
        
        # Load the image
        image = Image.open(data['image_path'])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.convert('RGB')
            image = pil_to_tensor(image)
            image = image / 255.
            image = image.to(self.device)
        
        # Extract the annotations
        annotations = torch.tensor(data['annotations'], dtype=torch.float32)
        
        # Move to the device
        image = image.to(self.device)
        annotations = annotations.to(self.device)
        
        return (image, annotations)