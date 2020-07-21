import json
import os.path as ops
import cv2
gt_root = '/data1/zem/Resnet.CRNN/data/ReCTS/gt_unicode'
file_name = 'train_ReCTS_000001.json'
with open(ops.join(gt_root,file_name),'r') as f:
    data = json.load(f)
for key, values in data.items():
    print(key,'=>',values)
img = cv2.imread('/data1/zem/Resnet.CRNN/data/ReCTS/img/train_ReCTS_000001.jpg')
for texts in data['lines']:
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
    x_max = max(x1,x2,x3,x4)
    y_max = max(y1,y2,y3,y4)
    x_min = min(x1,x2,x3,x4)
    y_min = min(y1,y2,y3,y4)
    left_top = (x_min,y_min)
    right_bottom = (x_max,y_max)
    cv2.rectangle(img,left_top,right_bottom,(0,0,255),2)

cv2.imwrite('/data1/zem/Resnet.CRNN/Lines.png',img)
print('finish')



