"""
paper: T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Lirong Zheng
date: 01/01/21
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            images_names = [i.strip() for i in contents]
            haze_names = images_names
            gt_names = images_names
            # haze_names = [i.strip() for i in contents]
            # gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        #self.images_names = images_names
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        # images_name = self.images_names[index]
        # imgs = Image.open(self.val_data_dir + images_name)
        # width, height = imgs.size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index][:4]+'.png'
        haze_img = Image.open(self.val_data_dir + 'haze/' +haze_name)
        gt_img = Image.open(self.val_data_dir + 'gt/' +gt_name)

        #haze_img = imgs.crop((0, 0, width // 2, height))
        #gt_img = imgs.crop((width // 2, 0, width, height))

        # haze_name = self.haze_names[index]
        # gt_name = self.gt_names[index]
        # haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        # gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
