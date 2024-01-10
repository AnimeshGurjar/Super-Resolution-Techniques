import random
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose

import sys
sys.path.append('/s/bach/a/class/cs435/cs435g/srcnn/model')

train_transformer = Compose([transforms.ToTensor()])
eval_transformer = Compose([transforms.ToTensor()])

class ImageDataset(Dataset):
    def __init__(self, blur_data_dir, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if self.is_supported_image(f)]

        self.blur_filenames = os.listdir(blur_data_dir)
        self.blur_filenames = [os.path.join(blur_data_dir, f) for f in self.blur_filenames if self.is_supported_image(f)]

        #print("Filtered Filenames:", self.filenames)
        #print("Filtered Blur Filenames:", self.blur_filenames)

        self.transform = transform

    def is_supported_image(self, file_path):
        supported_formats = {'.jpg', '.jpeg', '.png'}
        # Extract the file extension and check if it is in the supported formats
        _, file_extension = os.path.splitext(file_path)
        return file_extension.lower() in supported_formats

    def get_image_format(self, file_path):
        try:
            with Image.open(file_path) as img:
                format = img.format.lower()  # returns 'jpeg', 'png', etc.
                #print(f"File: {file_path}, Format: {format}")
                return format
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            return None

    def __len__(self):
        #print("Number of samples in the dataset:", len(self.filenames))
        return len(self.filenames)

    def __getitem__(self, idx):
        label_image = Image.open(self.filenames[idx])
        train_image = Image.open(self.blur_filenames[idx])

        # print(f"Label image path: {self.filenames[idx]}")
        # print(f"Train image path: {self.blur_filenames[idx]}")

        label_image = self.transform(label_image)
        train_image = self.transform(train_image)

        # Convert images to the same data type
        label_image = label_image.float()
        train_image = train_image.float()

        return train_image, label_image

def collate_fn(batch):
    # Resize or crop images to a consistent size
    target_size = (144, 144)  
    train_images, label_images = zip(*batch)

    # Ensure that both images have the same number of channels
    train_images = [transforms.functional.resize(img, target_size) for img in train_images]
    label_images = [transforms.functional.resize(img, target_size) for img in label_images]

    # Convert PIL Images to tensors
    train_images = [transforms.functional.to_tensor(img) if isinstance(img, Image.Image) else img for img in train_images]
    label_images = [transforms.functional.to_tensor(img) if isinstance(img, Image.Image) else img for img in label_images]

    # Ensure both images have the same number of channels
    train_images = [img.repeat(3, 1, 1) if img.shape[0] == 1 else img for img in train_images]
    label_images = [img.repeat(3, 1, 1) if img.shape[0] == 1 else img for img in label_images]

    # Convert images to the same data type
    train_images = [transforms.functional.convert_image_dtype(img, torch.float) for img in train_images]
    label_images = [transforms.functional.convert_image_dtype(img, torch.float) for img in label_images]

    # Stack tensors
    return torch.stack(train_images), torch.stack(label_images)

def fetch_dataloader(types, data_dir, params):  
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        if split in types:
            path_blur = os.path.join(data_dir, "{}_blur".format(split))
            path = os.path.join(data_dir, "{}_clear".format(split))

            if split == 'train':
                dl = DataLoader(
                    ImageDataset(path_blur, path, train_transformer),
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                    collate_fn=collate_fn  # Use the custom collate function
                )
            else:
                dl = DataLoader(
                    ImageDataset(path_blur, path, eval_transformer),
                    batch_size=params.batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                    collate_fn=collate_fn  # Use the custom collate function
                )
            dataloaders[split] = dl

    return dataloaders
