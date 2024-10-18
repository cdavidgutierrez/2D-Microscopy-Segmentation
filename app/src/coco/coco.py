import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple


class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, num_classes: int, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)
    
    def _create_colored_mask(mask, num_classes=4):
        # Definir colores para cada clase (en formato RGB)
        colors = {
            0: [0, 0, 0],       # Clase 0 - Fondo (Negro)
            1: [255, 0, 0],     # Clase 1 - Rojo
            2: [0, 255, 0],     # Clase 2 - Verde
            3: [0, 0, 255],     # Clase 3 - Azul
            4: [255, 255, 0]    # Clase 4 - Amarillo
        }
        
        # Asegúrate de que la máscara esté en formato numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Crear una máscara RGB
        height, width = mask.shape
        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        for cls in range(num_classes + 1):
            mask_rgb[mask == cls] = colors[cls]
        
        return mask_rgb

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[index]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        image_info = self.coco.loadImgs(img_id)[0]
        image_path = f"{self.root_dir}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        # Crear una máscara de segmentación con un canal por clase
        mask = np.zeros((self.num_classes, image_info['height'], image_info['width']), dtype=np.uint8)
        
        for ann in annotations:
            category_id = ann['category_id']
            if 'segmentation' in ann:
                rles = maskUtils.frPyObjects(ann['segmentation'], image_info['height'], image_info['width'])
                rle = maskUtils.merge(rles)
                m = maskUtils.decode(rle)
                mask[category_id] = np.maximum(mask[category_id], m)

        mask = torch.tensor(mask, dtype=torch.float32)
        mask_rgb = self._create_colored_mask(mask)

        if self.transform:
            image = self.transform(image)
            if not isinstance(mask, torch.Tensor):
                mask_rgb = self.transform(mask_rgb)

        return image, mask_rgb

transform = transforms.Compose([
    transforms.Resize((2048, 1728)),  # Tamaño ajustado
    transforms.ToTensor(),
])

# Número de clases en tu dataset
num_classes = 4

coco_dataset_service = COCOSegmentationDataset(
    root_dir='./data/train', 
    annotation_file='./data/train/_annotations.coco.json', 
    num_classes=num_classes,
    transform=transform
)

train_loader = DataLoader(coco_dataset_service, batch_size=1, shuffle=True)
