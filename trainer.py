import cv2
import os
import torch
import argparse
import torch.nn as nn
from utils import utils
import json
import string
import numpy as np
from datasets import concatdataset, dataset
from modules import resnet_aster
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

'''
annotation_train 7224612
annotation_test 891927 
annotation_val 802734 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot',required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to validation dataset')
parser.add_argument('--worker', type=int, help='number of data loading workers',default=4)
parser.add_argument('--trainbatchSize',type=int, default=800, help='the input batch size')
parser.add_argument('--testbatchSize',type=int, default=500, help='the input batch size')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--log_dir', default='log', help='Where to store log')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--trainNumber', type=int, default=100000000, help='Number of samples to train')
parser.add_argument('--testNumber', type=int, default=100000000, help='Number of samples to test')
parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--lr',type=float, default=1,help='learning rate')
parser.add_argument('--weight_decay',type=float, default=0.0)
parser.add_argument('--is_English',type=bool,default=False)
parser.add_argument('--pretrain',type=bool, default=True)



opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)


cudnn.benchmark = True

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),' is avaliable')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = torch.device('cuda:0')
else:
    print('using cpu actually')
    torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device('cpu')


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


def get_data(data_dir,flag, num, batch_size,worker,is_train):
    if isinstance(data_dir, list):
        assert len(data_dir) == len(flag), "the data dir don't match the flag"
        dataset_list = []
        for i in range(len(data_dir)):
            if flag[i] == 'E':
                dataset_list.append(dataset.LmdbDataset(data_dir[i],num=num,transform=dataset.resizeNormalize((100,32),is_train=is_train)))
            elif flag[i] == 'C':
                dataset_list.append(dataset.Chinese_LmdbDataset(data_dir[i],is_train=is_train,transform=dataset.resizeNormalize((100,32),is_train=is_train)))
            else:
                print('error flag %s' % (flag[i]))
        if is_train:
            datasets = concatdataset.ConcatDataset_V2(dataset_list)
        else:
            datasets = concatdataset.ConcatDataset(dataset_list)
    else:
        if flag == 'E':
            datasets = dataset.LmdbDataset(data_dir,num=num,transform=dataset.resizeNormalize((100,32),is_train=is_train))
        elif flag == 'C':
            datasets = dataset.Chinese_LmdbDataset(data_dir,is_train=is_train,transform=dataset.resizeNormalize((100,32),is_train=is_train))
        else:
            datasets = None
    print('total image', len(datasets))

    data_loader = DataLoader(datasets,batch_size=batch_size,shuffle=True,num_workers=worker,drop_last=True)

    return datasets, data_loader

if opt.is_English:
    train_dir_list = [opt.trainRoot+i for i in ['/CVPR2016','/NIPS2014']]
    test_dir = opt.trainRoot + '/benchmark_lmdbs_new/IIIT5K_3000'
    train_flag = []
    test_flag = []
else:
    train_dir_list = [opt.trainRoot+i for i in ['/ReCTS','/NIPS2014','/CVPR2016']]
    test_dir = [opt.trainRoot+i for i in ['/ReCTS','/benchmark_lmdbs_new/IIIT5K_3000']]
    train_flag = ['C','E','E']
    test_flag = ['C','E']

train_dataset, train_loader = get_data(train_dir_list,flag=train_flag, num=opt.trainNumber,batch_size=opt.trainbatchSize, worker=opt.worker,is_train=True)
test_dataset, test_loader = get_data(test_dir,flag=test_flag, num=opt.testNumber,batch_size=opt.testbatchSize, worker=opt.worker,is_train=False)


if opt.is_English:
    alphabet = string.printable[:-6]
    convert = dataset.strLabelToInt(alphabet)
    class_num = convert.num_class
else:
    with open('/data1/zem/Resnet.CRNN/alphabet.json','r') as f:
        data = json.load(f)
    alphabet = data['alphabet']
    convert = dataset.strLabelToInt(alphabet)
    class_num = convert.num_class

criterion = nn.CTCLoss(zero_infinity=True)
loss_avg_for_val = utils.Averager()
loss_avg_for_tra = utils.Averager()



crnn = resnet_aster.ResNet_ASTER(num_class=class_num,with_lstm=True).to(device)

param_groups = crnn.parameters()
param_groups = filter(lambda p: p.requires_grad, param_groups)
optimizer = torch.optim.Adadelta(param_groups, lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 50], gamma=0.1)


# net = torch.nn.DataParallel(crnn,device_ids=range(torch.cuda.device_count()))
criterion = criterion.to(device)


if opt.pretrain:
    crnn.load_state_dict(torch.load('/data1/zem/Resnet.CRNN/expr/best_model.pth', map_location='cuda:0'),strict=False)
    print('successfully load model')
    start_epoch = 0
else:
    start_epoch = 0


#record best model
best_model = None

def Val(net,data_loader,criterion,best_model,max_iter=1000000):
    print("Start Validation")
    is_change = False

    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    val_iter = iter(data_loader)
    sum_dist = 0
    max_iter = min(len(data_loader),max_iter)
    pbar = tqdm(range(max_iter))
    for i in pbar:
        pbar.set_description('running evaluation')
        data = val_iter.next()
        cpu_image, cpu_text = data
        image = cpu_image.to(device)
        Int_text,Int_length = convert.encoder(cpu_text)
        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * opt.testbatchSize)) #batch*[seq_len]
        cost = criterion(preds,Int_text,preds_size,Int_length)
        loss_avg_for_val.add(cost)
        _, preds = preds.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = convert.decoder(preds,preds_size)
        print('\n',"the predicted text is {}, while the real text is {}".format(sim_preds[0], cpu_text[0]))
        for pred, target in zip(sim_preds, cpu_text):
            dist = Lev_distance(pred, target)
            dist /= max(len(pred), len(target))
            sum_dist += dist
    res_dist = 1-float(sum_dist)/1000
    if best_model == None:
        best_distance = res_dist
        is_change = True
    else:
        if res_dist > best_model:
            best_distance = res_dist
            is_change = True
        else:
            best_distance = best_model


    print('Test loss: %f, Lev_distance: %f, best distance %f' % (loss_avg_for_val.val(), res_dist, best_distance))
    loss_avg_for_val.reset()
    return best_distance, is_change

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_image, cpu_text = data
    image = cpu_image.to(device)
    Int_text,Int_length = convert.encoder(cpu_text)
    assert len(Int_text) == Int_length.sum(), 'the encoded text length is not equal to variable length '
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * opt.trainbatchSize))  # batch*seq_len
    torch.backends.cudnn.enabled = False
    cost = criterion(preds, Int_text, preds_size, Int_length)
    torch.backends.cudnn.enabled = True
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

#start log
if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)
f_name = '{0}/synthesis_log_for_train.txt'.format(opt.log_dir)
f = open(f_name,'w')
for epoch in range(start_epoch,opt.nepoch):
    print('current epoch: %d' % (epoch+1))
    train_iter = iter(train_loader)
    tbar = tqdm(range(len(train_loader)))
    for i in tbar:
        tbar.set_description('running trainning')
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        cost = trainBatch(crnn,criterion,optimizer)
        loss_avg_for_tra.add(cost)
        if i%opt.displayInterval == 0 and i != 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch+1, opt.nepoch, i, len(train_loader), loss_avg_for_tra.val()))
            f.write('[%d/%d][%d/%d] Loss: %f \n' % (epoch+1, opt.nepoch, i, len(train_loader), loss_avg_for_tra.val()))
            loss_avg_for_tra.reset()
        if i%opt.valInterval == 0 and i != 0:
            best_model, is_change = Val(crnn,test_loader,criterion,best_model)
            if is_change:
                f.write('------------------------------------------------------\n')
                f.write('Best distance %f, the model name is RESNET_CRNN_%d_%d.pth \n' % (best_model,epoch+1,i))
                f.write('------------------------------------------------------\n')
                torch.save({'model':crnn.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},'{0}/synthesis_best_model.pth'.format(opt.expr_dir))
    scheduler.step()
    #logging finish
f.close()
print('training successfully finish')