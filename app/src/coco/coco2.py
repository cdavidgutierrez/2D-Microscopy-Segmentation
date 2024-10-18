import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from PIL import Image

class CocoImagesService():
    def __init__(self, root_dir: str, annotation_file: str):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.coco = COCO(root_dir + annotation_file)
        self.ids = list(self.coco.imgToAnns.keys())
        self.CATEGORY_COLORS = {
            1: (255, 0, 0),   # Rojo
            2: (0, 255, 0),   # Verde
            3: (0, 0, 255),   # Azul
            4: (255, 255, 0)  # Amarillo
        }

    def create_masks(self, prefix_name: str, file_path: str):
        image_ids = self.coco.getImgIds()
        for image_id in image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = image_info['file_name']
            height, width = image_info['height'], image_info['width']

            mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)

            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotation_ids)

            if not annotations:
                print(f"No annotations for image {image_id}")
                continue
            
            for annotation in annotations:
                category_id = annotation['category_id']
                segmentation = annotation['segmentation']

                if not segmentation:
                    print(f"No segmentation data for annotation {annotation['id']} in image {image_id}")
                    continue
                
                color = self._get_category_color(category_id)
                print(color)

                for polygon in segmentation:

                    if len(polygon) < 6:
                        print(f"Invalid polygon in annotation {annotation['id']} for image {image_id}")
                        continue
                    polygon = np.array(polygon).reshape(-1, 2)
                    polygon = np.clip(polygon, 0, [width - 1, height - 1])
                    polygon = polygon.astype(np.int32)
                    cv2.fillPoly(mask_rgb, [polygon], color)

            mask_image = Image.fromarray(mask_rgb)
            print(f'{self.root_dir}{file_path}{prefix_name}_{image_id}.png')
            mask_image.save(f'{self.root_dir}{file_path}{prefix_name}_{image_id}.png')
            #mask_image.save(f'data/masks/test/mask_{image_id}.png')

    def _get_category_color(self, category_id: int):
        return self.CATEGORY_COLORS.get(category_id, (255, 255, 255))
    
coco_dataset_service = CocoImagesService(root_dir='/home/user/Documentos/Estudiar/2D_Microscopy/2D-Segmentation/data/',
                                         annotation_file='test/_annotations.coco.json'
                                         )