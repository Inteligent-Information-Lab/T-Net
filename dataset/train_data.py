"""
paper: T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing
file: train_data.py
about: build the training dataset
author: Lirong Zheng
date: 01/01/21
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange, random
from torch import LongTensor, index_select
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, no_flip):
        super().__init__()
        train_list = train_data_dir+'train_list.txt'
        with open(train_list) as f:
            contents = f.readlines()
            images_names = [i.strip() for i in contents]
            haze_names = images_names
            gt_names = images_names
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        # Is flip true？
        self.no_flip = no_flip

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        #images_name = self.images_names[index]
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        # --- from image pairs gained hazy and gt image --- #
        haze_img = Image.open(self.train_data_dir + 'haze/' +haze_name)
        gt_img = Image.open(self.train_data_dir + 'gt/' +gt_name)        
        try:
            gt_img = Image.open(self.train_data_dir + 'gt/' + gt_name + '.jpg')
        except:
            gt_img = Image.open(self.train_data_dir + 'gt/' + gt_name + '.png').convert('RGB')

        width, height = haze_img.size


        #if width < crop_width or height < crop_height:
        #    raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        # Randomly cut
        # width = width // 2
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
       
        # --- Transform to tensor and Normalize--- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(haze_name))

        # --- If flip the picture --- #
        # Randomly flip
        if (not self.no_flip) and random() < 0.5:
            idx = [i for i in range(haze.size(2) - 1, -1, -1)]
            idx = LongTensor(idx)
            haze = haze.index_select(2, idx)
            gt = gt.index_select(2, idx)

        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

