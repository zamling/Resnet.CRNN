import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import json
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re


def crop_img(img,texts):
    x1 = texts['points'][0]
    y1 = texts['points'][1]
    x2 = texts['points'][2]
    y2 = texts['points'][3]
    x3 = texts['points'][4]
    y3 = texts['points'][5]
    x4 = texts['points'][6]
    y4 = texts['points'][7]
    '''
    1(23,173)
    2(327,180)
    3(327,290)
    4(23, 283)
    
    '''
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    crop_img = img[y_min:y_max,x_min:x_max]
    return crop_img



def checkImageIsValid(image):
    if image is None:
        return False
    imgH, imgW = image.shape[0], image.shape[1]
    if imgH * imgW == 0 or imgH > imgW:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def _is_difficult(word):
    assert isinstance(word, str)
    return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, jsonList):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(jsonList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        json_file = jsonList[i]
        img = cv2.imread(imagePath)

        with open(json_file, 'r') as f:
            data = json.load(f)

        label_list = []
        img_list = []
        for info in data['lines']:
            if info['ignore'] != 0:
                continue
            crop_image = crop_img(img,info)
            if checkImageIsValid(crop_image):
                label_list.append(info['transcription'])
                img_list.append(crop_image.tobytes())
            else:
                print('%s is invalid, the Height is %d, the Width is %d' % (imagePath,crop_image.shape[0],crop_image.shape[1]))
        assert len(label_list) == len(img_list),'the label_list != img_list'
        for i in range(len(img_list)):
            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = img_list[i]
            cache[labelKey] = label_list[i].encode()
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
    iters = 20000
    gt_root = '/data1/zem/Resnet.CRNN/data/ReCTS/gt_unicode'
    img_root = '/data1/zem/Resnet.CRNN/data/ReCTS/img'
    lmdb_output = '/data1/zem/Resnet.CRNN/data/lmdb/recognition/ReCTS'
    img_list = []
    json_list = []
    for i in range(iters):
        img_list.append(os.path.join(img_root,'train_ReCTS_%06d.jpg' % (i+1)))
        json_list.append(os.path.join(gt_root,'train_ReCTS_%06d.json' % (i+1)))
    createDataset(outputPath=lmdb_output,imagePathList=img_list,jsonList=json_list)






