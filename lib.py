import os
from typing import Dict

from PIL import Image
import numpy as np
from utils import ShapeRecogModel

class ShapeModel:
    def __init__(self, model_path: str, samples_root: str, invert: bool = False):
        """Create ShapeModel instance

        Args:
            model_path (str): path to pre-trained model
            samples_root (str): path to samples dir
            invert (bool, optional): invert the image from samples dir (0->1, 1->0) or not. Defaults to False.
        """
        self.model = ShapeRecogModel(model_path=model_path)
        self.samples_root = samples_root
        self.invert = invert
        
        self.samples_class, self.samples_feature = self.load_samples()
        
    @staticmethod
    def invert_image(image: Image) -> Image:
        np_image = np.array(image)
        invert_mask = np.ones(np_image.shape)*255
        invert_image = invert_mask - np_image   
        return Image.fromarray(np.uint8(invert_image)).convert('RGB') 
        
    def load_samples(self):
        samples = os.listdir(self.samples_root)
        samples_image = []
        for sample in samples:
            if self.invert is False:
                samples_image.append(Image.open(os.path.join(self.samples_root, sample)))
            else:
                samples_image.append(self.invert_image(Image.open(os.path.join(self.samples_root, sample)).convert('RGB')))
        
        
        samples_class = [sample.split('.')[0] for sample in samples]
        
        samples_feature = [self.model.extract_feature(np.array(image), do_extract=False) for image in samples_image]
        
        return samples_class, samples_feature 
    
    def reload_samples(self, path: str):
        """change samples dir

        Args:
            path (str): path to samples dir
        """
        self.samples_root = path
        self.samples_class, self.samples_feature = self.load_samples()
    
    @staticmethod
    def cosine_sim(emb1: np.array, emb2: np.array) -> float:
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1,emb2)
    
    def predict_emb(self, emb: np.array) -> str:
        result = dict()
        for name, feature in zip(self.samples_class, self.samples_feature):
            result[name] = self.cosine_sim(emb, feature)
            
        max_candidate = max(result.keys(), key= lambda x: result[x])
        
        return max_candidate
    
    def __call__(self, image: Image) -> Dict:
        """Do prediction

        Args:
            image (Image): Pillow Image, or numpy convertible image type

        Returns:
            Dict: dict(class=class_name)
        """
        emb = self.model.extract_feature(image = np.array(image), do_extract=False)
        
        class_name = self.predict_emb(emb)
        
        return {'class': class_name}
        