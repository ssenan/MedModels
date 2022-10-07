import os,sys
from re import I
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchio as tio


class ScanDataset(Dataset):
    def __init__(self, file_path, transform, PGGAN = False):
        super().__init__()
        self.image_path = str(file_path) + '/ct_train_image/'
        self.label_path = str(file_path) + '/ct_train_label/'
        self.image_list = [] 
        #self.label_list = sorted([f for f in os.listdir(self.label_path) if not f.startswith('.')]) 
        self.transform = transform
        self.PGGAN = PGGAN

        sort_image = sorted([f for f in os.listdir(self.image_path) if not f.startswith('.')])
        for image in sort_image:
            self.image_list.append(os.path.join(self.image_path,image))
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        '''
        # Loading the images and associated labels 
        curr_img = nib.load(self.image_path + self.image_list[index])
        curr_label = nib.load(self.label_path + self.label_list[index])

        # Loading the affine data for image and label   
        curr_img = curr_img.get_fdata()
        curr_label = curr_label.get_fdata()

        # Change size of image to desired size
        curr_img = resize(curr_img, [256,256,256], order=1, preserve_range=True)

        #Expand dimension of img
        curr_img = np.expand_dims(curr_img, axis=0)
        


        # Loading the files into tensors
        curr_img = torch.tensor(curr_img, dtype=torch.float32)
        curr_label = torch.tensor(curr_label)
        '''
        data_path = self.image_list[index]
        curr_img = tio.ScalarImage(data_path).data
        #curr_label = tio.ScalarImage(os.path.join(self.label_path, self.label_list[index])).data

        curr_img = torch.permute(curr_img, (0,3,1,2))

        if self.transform:
            curr_img = self.transform(curr_img)
            #curr_label = self.transform(curr_label)

        # Adding a channel value to the beginning of the img tensor
        #curr_img = torch.unsqueeze(curr_img, dim=0)
        # Labels not needed to train PGGAN
        return curr_img
        """
        if self.PGGAN:
            return curr_img
        else:
            return curr_img, #curr_label
        """

def get_dataset_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

def sample_data(dataloader, img_size):
    transform = transforms.Compose([
        transforms.Resize(2**img_size)
    ])
    loader = dataloader(transform)

    return loader

def image_loader(path):
    def loader(transform):
        data = ScanDataset(path, transform=transform)
        data_loader = DataLoader(
            data, 
            shuffle=True, 
            batch_size=1,
            num_workers=4
            )
        return data_loader
    return loader

def get_data(args):
    transform = tio.Compose([
        tio.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ScanDataset(args.dataset_path, transform=transform, PGGAN=True) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def get_data_noarg(dataset_path, batch_size):
    transform = tio.Compose([
        tio.transforms.Resize(64),  # args.image_size + 1/4 *args.image_size
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ScanDataset(dataset_path, transform=transform, PGGAN=True) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
