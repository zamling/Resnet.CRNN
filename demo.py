import torch
from torch.autograd import Variable
import utils
import string
import json
from datasets import dataset
from PIL import Image

from modules import resnet_aster


model_path = './expr/best_model.pth'
img_path = './data/demo.png'
is_english = True

if is_english:
    alphabet = string.printable[:-6]
    convert = dataset.strLabelToInt(alphabet)
    class_num = convert.num_class
else:
    with open('./alphabet.json','r') as f:
        data = json.load(f)
    alphabet = data['alphabet']
    convert = dataset.strLabelToInt(alphabet)
    class_num = convert.num_class

model = resnet_aster.ResNet_ASTER(num_class=class_num,with_lstm=True)
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location='cpu'))


transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('RGB')
image = transformer(image)
image = image.unsqueeze(0)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
sim_pred = convert.decoder(preds.data, preds_size.data)
print('%-20s' % (sim_pred))