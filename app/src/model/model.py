import os
from PIL import Image
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import functional as F
#Consultar VGG16/VGG19

from app.src.preprocess.preprocess import dataset

class SegmentationModel:
    def __init__(self, num_classes: int, pretrained: bool=True):
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.dataset = dataset

    def load_data(self, batch_size: int=4):
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train(self, num_epochs: int=10, verbose: bool = True):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            start_epoch_time = time.time()
            loader = tqdm(self.dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]") if verbose else self.dataloader
            for i, (images, masks) in enumerate(loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)['out']

                loss = self.criterion(outputs, masks)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if verbose:
                    if i % 10 == 0:  # Cada 10 lotes
                        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")
                
            epoch_duration = time.time() - start_epoch_time  # Fin del cronómetro por época
            avg_loss = running_loss / len(self.dataloader)

            if verbose:
                print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f} seconds. Average loss: {avg_loss:.4f}")
                
            if verbose:
                loader.set_postfix({"loss": avg_loss})

            print(f"Epoch [{epoch+1}/{num_epochs}], loss: {running_loss/len(self.dataloader)}")

    def predict(self, image_path):
        self.model.eval()

        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)['out']

        output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        return output
    
segmentation_model = SegmentationModel(num_classes=4)