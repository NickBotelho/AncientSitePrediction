import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

class AncientSiteDataset(Dataset):
    def __init__(self, csv, root_dir = "", transform = None):
        self.info = pd.read_csv(csv, sep = '\t')
        self.root_dir = [os.path.join(root_dir, i) for i in self.info.iloc[:, 5]]
        self.labels = torch.tensor(self.info.iloc[:, 4] > 0)
        self.transform = transform
    def __len__(self):
        return len(self.info)
    def __getitem__(self, idx):
        img_dir = self.root_dir[idx]
        image = Image.open(img_dir)
        # for detection: 1/0
        # for detection on confidence level: 3/2/0
        label = self.info.iloc[idx, 4]
        label = 1 if label > 0 else 0

        if self.transform is not None:
            image = self.transform(image)

        return {'image': image, 'label': label}