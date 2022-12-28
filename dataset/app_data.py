"""
paper: T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing
file: app_data.py
about: build the real-world dataset
author: Lirong Zheng
date: 01/01/21
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Validation/test dataset --- #
class AppData(data.Dataset):
    def __init__(self, app_data_dir):
        super().__init__()
        app_list = app_data_dir+'apply_list.txt'
        with open(app_list) as f:
            contents = f.readlines()
            images_names = [i.strip() for i in contents]
            haze_names = images_names
        self.haze_names = haze_names
        self.app_data_dir = app_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        # gt_name = self.gt_names[index][:4]+'.png'
        haze_img = Image.open(self.app_data_dir +haze_name)
        # gt_img = Image.open(self.val_data_dir + 'gt/' +gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        #gt = transform_gt(gt_img)

        return haze, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
