# Resnet.CRNN
This repository implements the CRNN with Resnet in pytorch.

## Todo List
- [x] release the Chinese Version
- [x] release the English Version
- [ ] Finetune the Handwritting Version

## Requirement
- Python3 (Python3.7 is recommended)
- PyTorch >= 1.0 (1.2 is recommended)
- torchvision from master
- matplotlib
- GCC >= 4.9 (This is very important!)
- OpenCV
- CUDA >= 9.0 (10.0 is recommended) 

## Train
If you want to train
```
bash train.sh
```

## Demo
the demo which run a image has been made
```
python demo
```

## Result
[**IIIT5K**]

|             | @Acc |  Lev-distance |
|:-------------:|:------:|:----:|
|  CRNN (English)  |  90.9000 |   -  | 
| CRNN (Chinese) |  90.6667   | 95.4388 | 

[**ReCTS**]

|             | @Acc |  Lev-distance |
|:-------------:|:------:|:----:|
|  CRNN (English)  |  - |   -  | 
| CRNN (Chinese) |  -  | 80.90 | 
