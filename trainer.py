import cv2
import os
import torch
import argparse
import torch.nn as nn
from utils import utils

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
parser.add_argument('--worker', type=int, help='number of data loading workers',default=0)
parser.add_argument('--batchSize',type=int, default=10, help='the input batch size')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--alphabet',type=str,default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--trainNumber', type=int, default=100, help='Number of samples to train')
parser.add_argument('--testNumber', type=int, default=100, help='Number of samples to test')
parser.add_argument('--valInterval', type=int, default=2, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=50000, help='Interval to be displayed')
opt = parser.parse_args()


if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)


cudnn.benchmark = True

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),' is avaliable')
    device = torch.device('cuda:1')
else:
    print('using cpu actually')
    device = torch.device('cpu')


def get_data(data_dir, num, batch_size,worker):
    if isinstance(data_dir, list):
        dataset_list = []
        for data_dir_ in data_dir:
            dataset_list.append(dataset.LmdbDataset(data_dir_,num,transform=dataset.resizeNormalize((100,32))))
        datasets = concatdataset.ConcatDataset(dataset_list)
    else:
        datasets = dataset.LmdbDataset(data_dir,num,transform=dataset.resizeNormalize((100,32)))
    print('total image', len(datasets))

    data_loader = DataLoader(datasets,batch_size=batch_size,shuffle=True,num_workers=worker,drop_last=True)

    return datasets, data_loader


train_dir_list = []
test_dir_list = []

train_dataset, train_loader = get_data(train_dir_list, num=opt.trainNumber,batch_size=opt.batchSize, worker=opt.worker)
test_dataset, test_loader = get_data(test_dir_list, num=opt.testNumber,batch_size=opt.batchSize, worker=opt.worker)

# train_dataset = dataset.LmdbDataset(root=opt.trainRoot,num=opt.trainNumber,transform=dataset.resizeNormalize((100,32)))
#
# train_loader = DataLoader(dataset=train_dataset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.worker,drop_last=True)
#
# test_dataset = dataset.LmdbDataset(root=opt.valRoot,num=opt.testNumber,transform=dataset.resizeNormalize((100,32)))


convert = dataset.strLabelToInt(opt.alphabet)
criterion = nn.CTCLoss()
loss_avg_for_val = utils.Averager()
loss_avg_for_tra = utils.Averager()



crnn = resnet_aster.ResNet_ASTER(num_class=len(opt.alphabet)+1,with_lstm=True).to(device)

param_groups = crnn.parameters()
param_groups = filter(lambda p: p.requires_grad, param_groups)
optimizer = torch.optim.Adadelta(param_groups, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)
# optimizer = torch.optim.Adadelta(crnn.parameters(), lr=1.0, weight_decay=0.0)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5], gamma=0.1)


# net = torch.nn.DataParallel(crnn,device_ids=range(torch.cuda.device_count()))
criterion = criterion.to(device)



def Val(net,dataset,criterion,max_iter=10000):
    print("Start Validation")

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = DataLoader(dataset=dataset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.worker,drop_last=True)
    val_iter = iter(data_loader)
    n_correct = 0
    max_iter = min(len(data_loader),max_iter)
    pbar = tqdm(range(max_iter))
    for i in pbar:
        pbar.set_description('running evaluation')
        data = val_iter.next()
        cpu_image, cpu_text = data
        image = cpu_image.to(device)
        Int_text,Int_length = convert.encoder(cpu_text)
        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * opt.batchSize)) #batch*[seq_len]
        cost = criterion(preds,Int_text,preds_size,Int_length)/opt.batchSize
        loss_avg_for_val.add(cost)
        _, preds = preds.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = convert.decoder(preds,preds_size,raw=False)
        if i%5 == 0 and i != 0:
            print('\n',"the predicted text is {}, while the real text is {}".format(sim_preds[0], cpu_text[0]))
        for pred, target in zip(sim_preds,cpu_text):
            if pred == target.lower():
                n_correct += 1
    accuracy = n_correct / float(max_iter * opt.batchSize)

    print('Test loss: %f, accuray: %f' % (loss_avg_for_val.val(), accuracy))
    loss_avg_for_val.reset()

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_image, cpu_text = data
    image = cpu_image.to(device)
    Int_text,Int_length = convert.encoder(cpu_text)
    assert len(Int_text) == Int_length.sum(), 'the encoded text length is not equal to variable length '
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * opt.batchSize))  # batch*seq_len
    cost = criterion(preds, Int_text, preds_size, Int_length) / opt.batchSize
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    scheduler.step(epoch)
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
            loss_avg_for_tra.reset()
        if i%opt.valInterval == 0 and i != 0:
            Val(crnn,test_dataset,criterion)

        if i%opt.saveInterval == 0 and i != 0:
            torch.save(crnn.state_dict(),'{0}/720w_RESNET_CRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch+1, i))