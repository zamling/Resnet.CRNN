import string
import torch
from torch.backends import cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import datasets
from datasets.dataset import LmdbDataset
from datasets.concatdataset import ConcatDataset
import modules.resnet_aster as model
import sys
import config

# globals_args = config.get_args(sys.argv[1:])




def get_data(data_dir, num, batch_size,worker):
    if isinstance(data_dir, list):
        dataset_list = []
        for data_dir_ in data_dir:
            dataset_list.append(LmdbDataset(data_dir_,num,transform=datasets.dataset.resizeNormalize((100,32))))
        dataset = ConcatDataset(dataset_list)
    else:
        dataset = LmdbDataset(data_dir,num,transform=datasets.dataset.resizeNormalize((100,32)))
    print('total image', len(dataset))

    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=worker,drop_last=True)

    return dataset, data_loader

def main(args):
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), ' is avaliable')
        device = torch.device('cuda:1')
    else:
        print('using cpu actually')
        device = torch.device('cpu')


    '''
    logging part
    '''

    '''
    Create data loaders
    '''
    train_dataset, train_dataloader = get_data(args.trainRoot, num=args.trainNumber, batch_size=args.batchSize, worker=args.worker)
    test_dataset, test_dataloader = get_data(args.testRoot, num=args.testNumber, batch_size=args.batchSize, worker=args.worker)


    '''
    create the alphabet
    '''
    alphabet = string.digits + string.ascii_lowercase
    convert = datasets.dataset.strLabelToInt(alphabet=alphabet)



    '''
    create model
    '''
    net = model.ResNet_ASTER(num_class=len(alphabet)+1,with_lstm=True).to(device)

    #Optimizer
    param_groups = net.parameters()
    param_groups = filter(lambda p: p.requires_grad, param_groups)
    optimizer = optim.Adadelta(param_groups, lr=args.lr, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4,5], gamma=0.1)

    #Trainer

    #evaluator

    #start training
    for epoch in range(args.nepoch):
        scheduler.step(epoch)
        #trainer
    print('Test with best model:')
    #evaluator
    #logging.close()






if __name__ == "__main__":
    alphabet = string.digits + string.ascii_lowercase
    convert = datasets.dataset.strLabelToInt(alphabet=alphabet)
    texts,lengthes = convert.encoder([string.digits + string.ascii_lowercase , string.digits + string.ascii_lowercase+'('+'-', 'aaaaaaabbbbbbvvvvvvv))))))'])
    texts = convert.decoder(texts,lengthes,raw=False)
    print(texts,lengthes)
    print(convert.dict)






