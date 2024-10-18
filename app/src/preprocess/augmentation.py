import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

class DataAugmentationService(Dataset):
    def __init__(self, images_paths: list[str], masks_paths: list[str]) -> None:
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transform = None

        self.transformations = {
            'h_flip': transforms.RandomHorizontalFlip(),
            'v_flip': transforms.RandomVerticalFlip(),
            'rotation': transforms.RandomRotation(90)
        }

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        print(self.images_paths[index], self.masks_paths[index])
        image = Image.open(self.images_paths[index]).convert("RGB")
        mask = Image.open(self.masks_paths[index]).convert("RGB")
        print(mask.mode)
        mask = mask.convert("RGB")

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        stacked = torch.cat([image, mask], dim=0)

        if self.transform:
            stacked = self.transform(stacked)
        
        image = stacked[:3, :, :]
        mask = stacked[3:, :, :]

        utils.save_image(mask, 'mask_exa.png')

        return image, mask
    
    def _set_transform(self, transformations: list):
        transformations_stack = [self.transformations[t] for t in transformations]
        self.transform = transforms.Compose([
            *transformations_stack
        ])

def image_mask_lists(coco_file: str, path: str):
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    images_paths = []
    masks_paths = []
    for dct in data['images']:
        images_paths.append(f"{path}test/{dct['file_name']}")
        masks_paths.append(f"{path}masks/test/mask_{dct['id']}.png")

    return images_paths, masks_paths

images_paths, masks_paths = image_mask_lists(
    coco_file='/home/user/Documentos/Estudiar/2D_Microscopy/2D-Segmentation/data/test/_annotations.coco.json',
    path='/home/user/Documentos/Estudiar/2D_Microscopy/2D-Segmentation/data/'
    )

data_augmentation_service = DataAugmentationService(images_paths, masks_paths)




        
