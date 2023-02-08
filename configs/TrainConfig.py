import os

from .BaseConfig import BaseConfig

class TrainConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.params = {
            "OUTPUT_DIR": "output/",
            "N_EPOCHS": 5,
            "BATCH_SIZE": 4,
            "BASE_LR": 0.000001,
            "LR_GAMMA": 0.1,
            "LR_SCHEDULING_ON": False,
            "N_CHANNELS": 3,
            "N_CLASSES": 1,
            "IMG_HEIGHT": 384,
            "IMG_WIDTH": 384,
            "MODEL_NAME": "UNet",
            "DET_LOSS_FN": "SmoothL1Loss",
            "CLASS_LOSS_FN": "MSELoss",
            "DET_COEFFICIENT": 1.0,
            "CLASS_COEFFICIENT": 0.01,
            "EVAL_ONLY": False,
            "SHUFFLE": True,
            "VERBOSE": 1
        }
        
        self.dataset_params = {
            "DATASET_DIR": "/home/myeghiaz/Storage/Datasets/LINZ-Real/GSD:0.250m_sample-size:384",
            "APPLY_TRAIN_TRANSFORMS": True
        }
        
        self.logging_params = {
            "LOG_DIR": os.path.join(self.params['OUTPUT_DIR'], "logs")
        }
        
    def __str__(self):
        self.logger.info("Config info")
        text = ""
        text += "-" * 80 + "\n"
        text += "TrainConfig:\n"
        
        for param in self.get_all_params():
            text += f"{param}: {self.params[param]}\n"
        
        text += "-" * 80
        return text
    
    def get_all_params(self):
        params = {}
        
        # General parameters
        for param in self.params:
            params[param] = self.params[param]
    
        # Dataset parameters
        for param in self.dataset_params:
            params[param] = self.dataset_params[param]
            
        # Logging parameters
        for param in self.logging_params:
            params[param] = self.logging_params[param]
        
        return params