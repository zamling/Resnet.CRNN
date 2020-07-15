import argparse

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


def get_args(sys_args):
    global_args = parser.parse_args(sys_args)
    return global_args