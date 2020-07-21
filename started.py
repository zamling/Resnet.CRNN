import lmdb
import six
from PIL import Image
root = '/data1/zem/Resnet.CRNN/data/lmdb/recognition/CVPR2016'

env = lmdb.open(root,max_readers=32,readonly=True)

item = 1
txn = env.begin()
img_key = b"image-%09d" % item

imgbuf = txn.get(img_key)
print(imgbuf)
buf = six.BytesIO()
buf.write(imgbuf)
buf.seek(0)
img = Image.open(buf).convert('RGB')
img.show()
