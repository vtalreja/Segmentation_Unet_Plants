from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import glob
import random
import torchvision.transforms.functional as TF


class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
       

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(256))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            angle=random.randint(-10,10)
            image=TF.rotate(image,angle)
            mask=TF.rotate(mask,angle)


        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # image=TF.normalize(image, mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        return image, mask


    def augmented_transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(256))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # image=TF.normalize(image, mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        return image, mask

    def __getitem__(self, idx):
        image=Image.open(self.image_paths[idx])
        mask=Image.open(self.target_paths[idx])
        # t_image,t_mask=self.augmented_transform(image,mask) # Uncomment this line if using augmented data and comment out the next line
        t_image,t_mask=self.transform(image,mask)
        t_mask = t_mask > 0
        t_mask = t_mask.float()
        return t_image, t_mask

    def __len__(self):
        return len(self.image_paths)


