"""
paper: T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing
file: demo.py
about: test the Stack T-Net on real-world images
author: Lirong Zheng
date: 01/01/21
"""

# --- Imports --- #
import os
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.app_data import AppData
from network.rec_model import RecModel
from utils.utils import application

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for Stack T-Net')
parser.add_argument('-recurrent_iter', help='Set the recurrent iter of the base model', default=3, type=int)
parser.add_argument('-updown_pairs', help='Set the pairs of upsampling and downsampling blocks', default=4, type=int)
parser.add_argument('-rdb_pairs', help='Setthe pairs of rdb blocks in the trunk road', default=3, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-app_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='SOTS', type=str)
args = parser.parse_args()

recurrent_iter = args.recurrent_iter
updown_pairs = args.updown_pairs
rdb_pairs = args.rdb_pairs
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
app_batch_size = args.app_batch_size
category = args.category
network_height = updown_pairs + 1
network_width = rdb_pairs * 2

print('--- Hyper-parameters for testing ---')
print('app_batch_size: {}\nrecurrent_iter: {}\nupdown_pairs: {}\nrdb_pairs: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'
      .format(app_batch_size, recurrent_iter, updown_pairs, rdb_pairs, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
app_data_dir = './data/demo/'

# --- Gpu device --- #
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
app_data_loader = DataLoader(AppData(app_data_dir), batch_size=app_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = RecModel(recurrent_iter=recurrent_iter, height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
net.load_state_dict(torch.load('./checkpoint/'+'{}_haze_best_{}_{}'.format(category, updown_pairs, rdb_pairs)))


# --- Use the evaluation model in testing --- #
net.eval()
print('--- Testing starts! ---')
start_time = time.time()
application(net, app_data_loader, device, category, save_tag=True)
end_time = time.time() - start_time
print('appliction time is {0:.4f}'.format(end_time))
