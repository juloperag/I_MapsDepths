import numpy as np
import torch
import cv2
import sys
import os
from PIL import Image
sys.path.append(os.path.abspath("./Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

class ModelDepthAnythingV2:
    def __init__(self, encoder, max_depth):
        #Creacion de instancia del modelo Depth AnythingV2
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.max_depth = max_depth
        self.model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
        self.model = self.model.to(DEVICE).eval()

    def predictionDepth(self, image):
        #Conversion de imagen a formato "Image"
        color_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = color_image.size
        # Predecir profundidar
        pred = self.model.infer_image(image)
        # Reescalar imagen y convertir
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)
        resized_pred = np.array(resized_pred)
        resized_pred = self.map_matrix(resized_pred, np.min(resized_pred), np.max(resized_pred), 4092, 0) 
        #Entregar prediccion 
        return resized_pred 
    
    def map_matrix(self, matrix, in_min, in_max, out_min, out_max):
        return (matrix - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
