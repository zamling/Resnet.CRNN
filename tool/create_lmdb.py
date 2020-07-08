import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re

errors= ['/2069/4/192_whittier_86389.jpg',
              '/2025/2/364_SNORTERS_72304.jpg',
              '/2013/2/370_refract_63890.jpg',
              '/1881/4/225_Marbling_46673.jpg',
              '/1863/4/223_Diligently_21672.jpg',
              '/1817/2/363_actuating_904.jpg',
              '/913/4/231_randoms_62372.jpg',
              '/869/4/234_TRIASSIC_80582.jpg',
              '/495/6/81_MIDYEAR_48332.jpg',
              '/368/4/232_friar_30876.jpg',
              '/275/6/96_hackle_34465.jpg',
              '/173/2/358_BURROWING_10395.jpg']

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def _is_difficult(word):
    assert isinstance(word, str)
    return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if len(label) == 0:
            continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
    data_dir = './data/mnt/ramdisk/max/90kDICT32px'
    lmdb_output_path = './data/MJSynText_lmdb_output'
    txt_file = data_dir + '/annotation_train.txt'
    print(txt_file)
    image_dir = data_dir
    with open(txt_file, 'r') as f:
        dirty_data = f.readlines()
    pbar = tqdm(range(len(dirty_data)))
    imagePathList = []
    labelList = []

    for i in pbar:
        pbar.set_description('creating the dataset')
        pri_data = dirty_data[i].strip().split()[0][1:]
        if pri_data not in errors:
            imagePathList.append(data_dir + pri_data)
            label = pri_data.split('/')[-1].split('.')[0].split('_')[-2]
            labelList.append(label)
    createDataset(lmdb_output_path, imagePathList, labelList)