"""
paper: T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing
file: train.py
about: main entrance for training the Stack T-Net
author: Lirong Zheng
date: 01/01/21
"""


# --- Imports --- #
import time
import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.train_data import TrainData
from dataset.val_data import ValData
from network.rec_model import RecModel
from utils.utils import to_psnr, to_ssim_skimage, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from utils.perceptual import LossNetwork
from utils.SSIM_loss import SSIM
plt.switch_backend('agg')


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for Stack T-Net')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=14, type=int)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-recurrent_iter', help='Set the recurrent iter of the base model', default=3, type=int)
parser.add_argument('-updown_pairs', help='Set the pairs of upsampling and downsampling blocks', default=4, type=int)
parser.add_argument('-rdb_pairs', help='Setthe pairs of rdb blocks in the trunk road', default=3, type=int)
parser.add_argument('-depth_rate', help='Set the depth rate of every row', default=16, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
# parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
recurrent_iter = args.recurrent_iter
updown_pairs = args.updown_pairs
rdb_pairs = args.rdb_pairs
depth_rate = args.depth_rate
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
no_flip = args.no_flip
category = 'SOTS'
network_height = updown_pairs + 1
network_width = rdb_pairs * 2

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nrecurrent_iter: {}\nupdown_pairs: {}\nrdb_pairs: {}\n'
      'depth_rate: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, recurrent_iter, updown_pairs, rdb_pairs, depth_rate, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
num_epochs = 1000
train_data_dir = './data/train/'
val_data_dir = './data/test/'

# --- Gpu device --- #
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,5,6,1"
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = RecModel(recurrent_iter=recurrent_iter, height=network_height, width=network_width, depth_rate=depth_rate,num_dense_layer=num_dense_layer, growth_rate=growth_rate)


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

# mse_loss_net = nn.MSELoss()
# ssim_loss_net = SSIM()
# ssim_loss_net.to(device)
# ssim_loss_net.eval()
per_loss_net = LossNetwork(vgg_model)
per_loss_net.eval()


# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('./checkpoint/'+'{}_haze_best_{}_{}'.format(category, updown_pairs, rdb_pairs)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, no_flip), batch_size=train_batch_size, shuffle=True, num_workers=24)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Previous PSNR and SSIM in testing --- #
net.eval()
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

for epoch in range(num_epochs):
    psnr_list = []
    ssim_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):
        haze, gt = train_data
        haze = haze.to(device)
        gt = gt.to(device) 

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        dehaze, dehaze_list = net(haze)
        smooth_loss = 0
        perceptual_loss = 0
        for i in dehaze_list:
            smooth_loss = smooth_loss + F.smooth_l1_loss(i, gt) 
            perceptual_loss = perceptual_loss + per_loss_net(i, gt)


        # --- Calculate the whole loss ---#
        # smooth_loss = F.smooth_l1_loss(dehaze, gt) 
        # mse_loss = mse_loss_net(dehaze, gt)
        # ssim_loss = - ssim_loss_net(dehaze, gt)
        # perceptual_loss = per_loss_net(dehaze, gt)
        # loss = ssim_loss
        loss = smooth_loss + lambda_loss*perceptual_loss


        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))
        ssim_list.extend(to_ssim_skimage(dehaze, gt))
        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)
    train_ssim = sum(ssim_list) / len(ssim_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), './checkpoint/'+'{}_haze_{}_{}'.format(category, updown_pairs, rdb_pairs))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, category)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, train_ssim, val_psnr, val_ssim, category)

    # --- update the network weight --- #
    if (val_psnr >= old_val_psnr):
        torch.save(net.state_dict(), '{}_haze_best_{}_{}'.format(category, updown_pairs, rdb_pairs))
        old_val_psnr = val_psnr
