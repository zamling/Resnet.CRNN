from modules.resnet_aster import ResNet_ASTER
import torch
import os
import torch
import argparse
import torch.nn as nn
from utils import utils
import string
import numpy as np
from datasets import concatdataset, dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
import json
import os




os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def Lev_distance(A, B):
    dp = np.array(np.arange(len(B) + 1))
    for i in range(1, len(A) + 1):
        temp1 = dp[0]
        dp[0] += 1
        for j in range(1, len(B) + 1):
            temp2 = dp[j]
            if A[i - 1] == B[j - 1]:
                dp[j] = temp1
            else:
                dp[j] = min(temp1, min(dp[j - 1], dp[j])) + 1
            temp1 = temp2
    return dp[len(B)]

with open('/data1/zem/Resnet.CRNN/alphabet.json', 'r') as f:
    data = json.load(f)
alphabet = data['alphabet']
convert = dataset.strLabelToInt(alphabet)
class_num = convert.num_class
test_data_dir = '/data1/zem/OCR_and_STR/data/lmdb/recognition/ReCTS'
datasets = dataset.Chinese_LmdbDataset(test_data_dir,is_train=False,transform=dataset.resizeNormalize((100,32),is_train=False))
test_loader = DataLoader(datasets,batch_size=500,shuffle=True,num_workers=0,drop_last=True)
print('the test loader has %d steps' % (len(test_loader)))
convert = dataset.strLabelToInt(alphabet)
#加载模型
device = torch.device('cpu')
model = ResNet_ASTER(num_class=convert.num_class,with_lstm=True).to(device)
# checkpoint = torch.load('/data1/zem/Resnet.CRNN/expr/chinese_best_model.pth',map_location='cuda:0')
model.load_state_dict(torch.load('/data1/zem/Resnet.CRNN/expr/chinese_best_model.pth',map_location='cpu'))


n_correct = 0

for p in model.parameters():
    p.requires_grad = False
model.eval()
val_iter = iter(test_loader)
tpar = tqdm(range(len(test_loader)))
sum_dist = 0
for i in tpar:
    data = val_iter.next()
    cpu_image, cpu_text = data
    image = cpu_image.to(device)
    Int_text,Int_length = convert.encoder(cpu_text)
    preds = model(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * 500)) #batch*[seq_len]
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = convert.decoder(preds,preds_size)
    if i%10 == 0 and i != 0:
        print('\n',"the predicted text is {}, while the real text is {}".format(sim_preds[0].lower(), cpu_text[0].lower()))
    for pred, target in zip(sim_preds,cpu_text):
        dist = Lev_distance(pred, target)
        dist /= max(len(pred),len(target))
        sum_dist += dist
print(sum_dist)
print('the distance is %f' % (1-float(sum_dist)/1000))

