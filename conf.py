import cv2
import torch
import numpy as np

device = 'cuda'  # 'cpu' or 'cuda'
test_path =r'./testdata/test1.jpg'
result_path =r'./result/result1.jpg'
lr = 1e-4
iterations = 1000
coarse_model_path = './checkpoint/coarse_final.pth'
fine_model_path = './checkpoint/fine_final.pth'
TV_factor = 0.1
exp_factor = 200
noise_factor = 5000
rec_factor = 1
vgg_factor = 1
detail_factor =2
channel = 16
channel_d = 32
illu_channel = 1
rgb_channel = 3
eps = 1e-6
contact_channel=6
