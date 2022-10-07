import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms

from skimage.transform import resize
import nibabel as nib
from pathlib import Path

def load_data_pairs_np(pair_list):
    # Loading all image and label pairs into lists
    img_list = []
    label_list = []
    data_dir = Path('./ct_train')
    for k in range(0, len(pair_list), 2):
        # Loading images and labels
        img_path = pair_list[k]
        label_path = pair_list[k + 1]
        img_data = nib.load(data_dir / img_path)
        label_data = nib.load(data_dir / label_path)

        # Getting affine data after resizing
        img_data = img_data.get_fdata()
        label_data = label_data.get_fdata()

        # Resizing images to (256 x 256 x 256)
        img_data = resize(img_data, (256,256,256))
        label_data = resize(label_data, (256,256,256))

        # Returning to Tensor
        img_data = torch.tensor(img_data)
        label_data = torch.tensor(label_data)

    return (img_data, label_data)
