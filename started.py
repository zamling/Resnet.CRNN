import lmdb
import torch
import chardet
from datasets import dataset
import cv2
import numpy as np
import six
from PIL import Image
import string
import json
from tqdm import tqdm
import os
from torch.utils import data
from torchvision import transforms


lmdb_path = '/data1/zem/Resnet.CRNN/data/lmdb/recognition/ReCTS'
train_dataset = dataset.LmdbDataset(root=lmdb_path,num=1000000,transform=dataset.resizeNormalize((100,32)))
batch_size = 1
train_dataloader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)

print('train_dataloader is %d' % (len(train_dataloader)))
for i, (images, labels) in enumerate(train_dataloader):
    toPILImage = transforms.ToPILImage()
    for id, (image, label) in enumerate(zip(images, labels)):
        print('this is the %d image' % (id))
        image = toPILImage(image)
        image.show()
        print(image.size)
        print(label)
        input('checkpoint')





