import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image

from app.src.coco.coco2 import coco_dataset_service
from app.src.model.model import SegmentationModel, segmentation_model
from app.src.preprocess.augmentation import data_augmentation_service

########### COCO FILE TO MASKS.

#coco_dataset_service.create_masks('mask', 'masks/test/')
#coco_dataset_service.create_masks('mask', 'masks/train/')
#coco_dataset_service.create_masks('mask', 'masks/valid/')

########### COCO FILE TO MASKS.



########### MODEL BUILDING AND TEST.

#model = SegmentationModel(num_classes=4)
#model.load_data(batch_size=4)
#model.train(num_epochs=1)

#segmentation_model.load_data(batch_size=4)
#segmentation_model.train(num_epochs=1)

# image_path = './data/valid/220324-hBN-on-100nm-SiO2-SP-1_23_jpg.rf.8e7687c49f55f34113512cca0f187ee9.jpg'  # Reemplaza con la ruta de tu imagen
# predicted_mask = segmentation_model.predict(image_path)

# plt.imshow(predicted_mask)
# plt.title('Predicted Segmentation Mask')
# plt.show()

########### MODEL BUILDING AND TEST.



########### TEST DATA AUGMENTATION.

data_augmentation_service._set_transform(
    transformations=['h_flip', 'v_flip']
    )
image, mask = data_augmentation_service[0]
mask.to
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title("Imagen transformada")
plt.subplot(1, 2, 2)
plt.imshow(mask[0], cmap='rainbow')
plt.title("Máscara transformada")
plt.show()

########### TEST DATA AUGMENTATION.