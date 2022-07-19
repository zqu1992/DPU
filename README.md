# DPU

Deep Partial Updating: Towards Communication Efficient Updating for On-device Inference

### Introduction
This repository contains the code of DPU introduced in our ECCV2022 paper:

Z. Qu, C. Liu, and L. Thiele. Deep Partial Updating: Towards Communication Efficient Updating for On-device Inference.  

### Dependencies

+ Python 3.9+
+ PyTorch 1.10.0+
+ NVIDIA GPU + CUDA CuDNN (CUDA 11.0)

### Usage

The MNIST and CIFAR10/100 datasets can be automatically downloaded via Pytorch.

ImageNet dataset should be downloaded and decompressed into the structure like,

    dir/
      train/
        n01440764/
          n01440764_10026.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...
You may follow some instructions provided in https://pytorch.org/docs/1.1.0/_modules/torchvision/datasets/imagenet.html
    
    
For more options, please refer to `python xxx.py -h` respectively.


### Citation
If you use the code in your research, please cite as

Zhongnan Qu, Cong Liu, Lothar Thiele. Deep Partial Updating: Towards Communication Efficient Updating for On-device Inference. In *the European Conference on Computer Vision* (ECCV), 2022.

    @InProceedings{Qu_2022_ECCV,
        author = {Qu, Zhongnan and Liu, Cong and Thiele, Lothar},
        title = {Deep Partial Updating: Towards Communication Efficient Updating for On-device Inference},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        month = {October},
        year = {2022}
    }

