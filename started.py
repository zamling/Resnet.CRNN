import lmdb
import torch
import chardet
import cv2
import numpy as np
import six
from PIL import Image
import string
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_root = '/home/ailab/data1/zem/Resnet.CRNN/expr'
best_model = 'RESNET_CRNN_7_10000.pth'
best_path = os.path.join(model_root,best_model)
checkpoint = torch.load(best_path)
model_pth = checkpoint['model']
torch.save(model_pth,'/home/ailab/data1/zem/Resnet.CRNN/expr/best_model.pth')
print('finish')





