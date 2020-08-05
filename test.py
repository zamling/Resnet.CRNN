from modules.resnet_aster import ResNet_ASTER
import torch
import os
import torch
import argparse
import torch.nn as nn
from utils import utils
import string
from datasets import concatdataset, dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
import json
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with open('/data1/zem/Resnet.CRNN/alphabet.json', 'r') as f:
    data = json.load(f)
alphabet = data['alphabet']
convert = dataset.strLabelToInt(alphabet)
class_num = convert.num_class
test_data_dir = '/data1/zem/OCR_and_STR/data/lmdb/recognition/ReCTS'
datasets = dataset.LmdbDataset(test_data_dir,1000000,transform=dataset.resizeNormalize((100,32)))
test_loader = DataLoader(datasets,batch_size=800,shuffle=True,num_workers=2,drop_last=True)
print('the test loader has %d steps' % (len(test_loader)))
convert = dataset.strLabelToInt(alphabet)
#加载模型
device = torch.device('cuda:0')
model = ResNet_ASTER(num_class=convert.num_class,with_lstm=True).to(device)
checkpoint = torch.load('/data1/zem/Resnet.CRNN/expr/Chinese_best_model.pth',map_location='cuda:0')
model.load_state_dict(checkpoint['model'])

n_correct = 0

for p in model.parameters():
    p.requires_grad = False
model.eval()
val_iter = iter(test_loader)
tpar = tqdm(range(len(test_loader)))
for i in tpar:
    data = val_iter.next()
    cpu_image, cpu_text = data
    image = cpu_image.to(device)
    Int_text,Int_length = convert.encoder(cpu_text)
    preds = model(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * 800)) #batch*[seq_len]
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = convert.decoder(preds,preds_size)
    if i%10 == 0 and i != 0:
        print('\n',"the predicted text is {}, while the real text is {}".format(sim_preds[0].lower(), cpu_text[0].lower()))
    for pred, target in zip(sim_preds,cpu_text):
        if pred == target:
            n_correct += 1
accuracy = n_correct / float(len(test_loader)*800)
print('the accuracy is %f' % accuracy)