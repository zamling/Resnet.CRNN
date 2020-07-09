import scipy.io as sio
from tqdm import tqdm
import cv2
import numpy as np
import os


matfile = '/data1/zem/Resnet.CRNN/data/SynthText/gt.mat'
img_root = '/data1/zem/Resnet.CRNN/data/SynthText'
data = sio.loadmat(matfile)

words = []
for val in data['txt'][0][0]:
    v = [x.split('\n') for x in val.strip().split(' ')]
    v = [[vv_ for vv_ in vv if len(vv_) > 0] for vv in v]
    words.extend(sum(v, []))
wordsBB = np.around(np.array(data['wordBB'][0][0]), decimals=2).transpose(2, 1, 0)
print(words[0])
x1 = wordsBB[0][0][0]
y1 = wordsBB[0][0][1]
x2 = wordsBB[0][1][0]
y2 = wordsBB[0][1][1]
x3 = wordsBB[0][2][0]
y3 = wordsBB[0][2][1]
x4 = wordsBB[0][3][0]
y4 = wordsBB[0][3][1]
img_path = data['imnames'][0][0][0]

img_abs_path = os.path.join(img_root,img_path)
print(img_abs_path)

img = cv2.imread(img_abs_path)
cv2.rectangle(img,(x1,y1),(x3,y3),(0,0,255),2)
cv2.imwrite('/data1/zem/Resnet.CRNN/Lines.png',img)

print('done')





'''
import scipy.io as sio

data = sio.loadmat(matfile)

    image_lists = []
    if not os.path.exists(GT_DIR):
        os.makedirs(GT_DIR)

    for i in tqdm(range(len(data['txt'][0]))):
        img_path = data['imnames'][0][i][0]
        image_lists.append(img_path)
        img_dir, img_name = img_path.split('/')
        if not os.path.exists(os.path.join(GT_DIR, img_dir)):
            os.makedirs(os.path.join(GT_DIR, img_dir))

        gt_dict = []
        words = []
        for val in data['txt'][0][i]:
            v = [x.split('\n') for x in val.strip().split(' ')]
            v = [[vv_ for vv_ in vv if len(vv_) > 0] for vv in v]
            words.extend(sum(v, []))
        chars = ''.join(words)
        if len(chars) == 0:
            continue

        wordsBB = np.around(np.array(data['wordBB'][0][i]), decimals=2)
        charsBB = np.around(np.array(data['charBB'][0][i]), decimals=2)

        if len(wordsBB.shape) == 3:
            wordsBB = wordsBB.transpose(2, 1, 0)
        else:
            wordsBB = wordsBB.transpose(1, 0)[np.newaxis, :]
        if len(charsBB.shape) == 3:
            charsBB = charsBB.transpose(2, 1, 0)
        else:
            charsBB = charsBB.transpose(1, 0)[np.newaxis, :]
        assert len(words) == len(wordsBB) and len(chars) == len(charsBB)

        char_offset = 0
        for j in range(len(wordsBB)):
            ins_gt = dict()
            charline = ''
            if 0 == len(words[j]):
                continue
            x1 = wordsBB[j][0][0]
            y1 = wordsBB[j][0][1]
            x2 = wordsBB[j][1][0]
            y2 = wordsBB[j][1][1]
            x3 = wordsBB[j][2][0]
            y3 = wordsBB[j][2][1]
            x4 = wordsBB[j][3][0]
            y4 = wordsBB[j][3][1]
            boxline = words[j]
            '''


