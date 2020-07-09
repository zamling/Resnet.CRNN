from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import lmdb
import cv2
import sys
import six
import torch
from torch.utils import data
import collections
from torch.utils.data import sampler
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LmdbDataset(data.Dataset):
    def __init__(self, root,num,transform = None):
        self.env = lmdb.open(root,max_readers=32,readonly=True)
        self.num = num
        assert self.env is not None, "cannot create the lmdb from %s" %root
        self.txn = self.env.begin()
        self.transform = transform
        self.nSamples = int(self.txn.get(b"num-samples"))
        self.nSamples = min(self.nSamples,num)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, item):
        item += 1
        img_key = b"image-%09d" % item
        imgbuf = self.txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')
        except IOError:
            print('Corrupted image for %d' % item)
            return self[item + 1]
        label_key = b'label-%09d' % item
        word = self.txn.get(label_key).decode()
        if self.transform is not None:
            img = self.transform(img)
        return img, word

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class strLabelToInt(object):
    def __init__(self, alphabet, is_ignore = True):
        self._is_ignore = is_ignore
        if self._is_ignore:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i,char in enumerate(alphabet):
            self.dict[char] = i + 1 # the '-' = 0

    def encoder(self,words):
        if isinstance(words,str):
            text = [self.dict[char.lower() if self._is_ignore else char] for char in words] #text is a list in Int
            length = len(text)
        elif isinstance(words,collections.Iterable):
            length = [len(s) for s in words]
            text = ''.join(words)
            text ,_ = self.encoder(text)
        return (torch.IntTensor(text),torch.IntTensor(length))

    def decoder(self,text,length,raw = True):
        if length.numel() == 1:
            length = length.item()
            assert length == text.numel(),"text has the length {}, while the claimed length is {}".format(text.numel(),length)
            if raw:
                str_text = ''.join([self.alphabet[i-1] for i in text]) # 0-1 = -1 最后一个是'-'
                return str_text
            else:
                char_list = []
                for i in range(length):
                    if text[i] != 0 and (not(i>0 and text[i-1] == text[i])):
                        char_list.append(self.alphabet[text[i]-1])
                return ''.join(char_list)
        else:
            assert text.numel() == length.sum(),"the batch text has the length {}, while the claimed length is {}".format(text.numel(),length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decoder(text[index:index+l],length[i],raw = raw)
                )
                index += l
            return texts