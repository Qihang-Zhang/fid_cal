import os
import torch
import pickle
import requests
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, transforms
from bidict import bidict
from tqdm import tqdm

my_bidict = bidict({'Class0': 0, 
                    'Class1': 1,
                    'Class2': 2,
                    'Class3': 3})

class SamplingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List to store image paths along with domain and category
        # Convert DataFrame to a list of tuples
        self.samples = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
        self.samples = [(os.path.join(self.root_dir, filename), 'None') for filename in self.samples]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, category = self.samples[idx]
        if category in my_bidict.values():
            category_name = my_bidict.inverse[category]
        else:
            category_name = "Unknown"
        # print(img_path)
        image = read_image(img_path)  # Reads the image as a tensor
        image = image.type(torch.float32) / 255.  # Normalize to [0, 1]
        if image.shape[0] == 1:
            image = replicate_color_channel(image)
        if self.transform:
          image = self.transform(image)
        return image, category_name
    
    def get_all_images(self, label):
        return [img for img, cat in self.samples if cat == label]

def show_images(images, categories):
        fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
        for i, image in enumerate(images):
            axs[i].imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            axs[i].set_title(f"Category: {categories[i]}")
            axs[i].axis('off')
        plt.savefig('_test.png')

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo)
    return data_dict

def download_and_extract(url, target_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*1024*10  # 1 Kilobyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        # Using system unzip to extract files
        subprocess.run(['unzip', '-q', target_path, '-d', os.path.dirname(target_path)], check=True)
        os.remove(target_path)  # Remove the zip file after extraction

class ImageNet32Dataset(Dataset):
    def __init__(self, data_folder, train=True, download=False, transform=None):
        self.data_folder = data_folder
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []

        if not os.path.exists(self.data_folder) and download:
            os.makedirs(self.data_folder)
            print("Downloading ImageNet 32x32 data...")
            download_and_extract('https://www.image-net.org/data/downsample/Imagenet32_train.zip', os.path.join(self.data_folder, 'Imagenet32_train.zip'))
            download_and_extract('https://www.image-net.org/data/downsample/Imagenet32_val.zip', os.path.join(self.data_folder, 'Imagenet32_val.zip'))


        if self.train:
            for i in range(1, 11):  # Assuming there are 10 batches as per your directory listing
                batch_data = self.load_databatch(i)
                self.data.append(batch_data['X_train'])
                self.labels.append(batch_data['Y_train'])
            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            batch_data = self.load_databatch(0, train=False)
            self.data = batch_data['X_train']
            self.labels = batch_data['Y_train']

    def load_databatch(self, idx, train=True):
        if train:
            data_file = os.path.join(self.data_folder, f'train_data_batch_{idx}')
        else:
            data_file = os.path.join(self.data_folder, 'val_data')
        
        d = unpickle(data_file)
        x = d['data']
        y = d['labels']
        # mean_image = d['mean']
        # print(mean_image.shape)

        x = x / np.float32(255)
        # mean_image = mean_image / np.float32(255)
        y = [i-1 for i in y]
        # x -= mean_image

        img_size2 = 32 * 32
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

        # Handling of mirrored images can be optional or handled as augmentation during training
        X_train = x
        Y_train = np.array(y, dtype='int32')

        return {'X_train': X_train.transpose((0, 2, 3, 1)), 'Y_train': Y_train}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    
    ds_transforms = transforms.Compose([rescaling])
        
    dataset = SamplingDataset(root_dir='./Sampling', transform=ds_transforms)
    data_loader = DataLoader(dataset, batch_size = 16, shuffle=True)
    # Sample from the DataLoader
    for images, categories in tqdm(data_loader):
        print(images.shape, categories)
        pdb.set_trace()
        images = torch.round(rescaling_inv(images) * 255).type(torch.uint8)
        show_images(images, categories)
        # break  # We only want to see one batch of 4 images in this example
        