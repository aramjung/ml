import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DigitDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index].reshape(28, 28).astype(np.uint8)
        img = Image.fromarray(img)

        img = self.transform(img) # apply augmentation

        if self.labels is not None:
            return img, self.labels[index]
        else:
            return img

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32 filters, filter size 3, padding 1: input 1*28*28, output 32*28*28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # input: 32*28*28, output: 32*14*14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # input: 32*14*14, output: 64*14*14
            nn.BatchNorm2d(64), # normalize
            nn.ReLU(),
            nn.MaxPool2d(2),  # intput: 64*14*14, output: 64*7*7

            nn.Flatten(),  # input: 64*7*7, output: 64*7*7 = 3136
            nn.Dropout(0.2), 
            nn.Linear(64 * 7 * 7, 128),     #input: 3136, output: 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 10)         #input: 128, output: 10 
        )

    def forward(self, x):
        return self.net(x)

