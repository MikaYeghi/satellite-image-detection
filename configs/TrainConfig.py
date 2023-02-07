from .BaseConfig import BaseConfig

class TrainConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.params = {
            "N_EPOCHS": 5,
            "BATCH_SIZE": 32,
            "BASE_LR": 0.001,
            "N_CHANNELS": 3,
            "N_CLASSES": 1,
            "IMG_HEIGHT": 384,
            "IMG_WIDTH": 384,
            "MODEL_NAME": "UNet",
            "LOSS_FN": "MSELoss",
            "EVAL_ONLY": False
        }
        
        self.dataset_params = {
            "DATASET_DIR": "/home/myeghiaz/Storage/Datasets/LINZ-Real/GSD:0.250m_sample-size:384",
            "APPLY_TRAIN_TRANSFORMS": True
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
        
        return params