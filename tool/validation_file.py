import hashlib
import os,sys

def CalcSha1(filepath):
    with open(filepath,'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()
        print(hash)
        return hash

if __name__ == "__main__":
    filepath = "/data1/zem/Resnet.CRNN/data/ReCTS.zip"
    print(CalcSha1(filepath))