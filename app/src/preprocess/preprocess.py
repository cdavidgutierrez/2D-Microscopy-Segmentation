import os
import json
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import functional as F

class PreprocessedSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder: str, mask_folder: str, coco_annotations_file: str):
        self.images_folder = images_folder
        self.masks_folder = mask_folder
        with open(coco_annotations_file, 'r') as f:
            coco_data = json.load(f)

        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        self.image_id_to_mask_filename = {
            img_id: self._get_mask_filename(img_id) for img_id in self.image_id_to_filename.keys()
        }

        self.CATEGORY_COLORS = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,   # Rojo
            (0, 255, 0): 2,   # Verde
            (0, 0, 255): 3,   # Azul
            (255, 255, 0): 4  # Amarillo
        }

    def _get_mask_filename(self, img_id):
        for mask_filename in os.listdir(self.masks_folder):
            if str(img_id) in mask_filename:
                return mask_filename
        raise FileNotFoundError(f"No se encontró máscara para la imagen con ID {img_id}")
        
    def __len__(self):
        return len(self.image_id_to_filename)
        
    def __getitem__(self, index):
        # Obtener el ID de la imagen según el índice
        img_id = list(self.image_id_to_filename.keys())[index]

        # Obtener la ruta de la imagen
        image_filename = self.image_id_to_filename[img_id]
        image_path = os.path.join(self.images_folder, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Obtener la ruta de la máscara
        mask_filename = self.image_id_to_mask_filename[img_id]
        mask_path = os.path.join(self.masks_folder, mask_filename)
        mask = Image.open(mask_path).convert("RGB")

        # Convertir la imagen y la máscara a tensores
        image = F.to_tensor(image)
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.long)
        for rgb, class_value in self.CATEGORY_COLORS.items():
            class_mask[(mask == rgb).all(axis=-1)] = class_value

        class_mask = torch.tensor(class_mask, dtype=torch.long)

        return image, class_mask
        
dataset = PreprocessedSegmentationDataset(
    images_folder='/home/user/Documentos/Estudiar/2D_Microscopy/2D-Segmentation/data/test/',
    mask_folder='/home/user/Documentos/Estudiar/2D_Microscopy/2D-Segmentation/data/masks/test/',
    coco_annotations_file='/home/user/Documentos/Estudiar/2D_Microscopy/2D-Segmentation/data/test/_annotations.coco.json'
)