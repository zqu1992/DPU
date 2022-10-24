# DPU

Deep Partial Updating: Towards Communication Efficient Updating for On-device Inference

### Introduction
This repository contains the code of DPU introduced in our ECCV2022 paper:

Z. Qu, C. Liu, and L. Thiele. Deep Partial Updating: Towards Communication Efficient Updating for On-device Inference.  

[PDF](https://arxiv.org/pdf/2007.03071.pdf)

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

For example, to update the VGGNet in multi-round on CIFAR10 dataset,

(1) Deep Partial Updating (DPU) with updating ratio 0.01, lambda 0.5 (see Appendix G.2), dataset size of 1000 before the initial deployment, dataset size of 1000 in each subsequent round, 

    python multiround_dpu.py --mode dpu --updating_ratio 0.01 --lamda 0.5 --model vgg --dataset cifar10 --dataset_size_init 1000 --dataset_size_subseq 1000

(2) Global Contribution Partial Updating (GCPU) with updating ratio 0.01, dataset size of 1000 before the initial deployment, dataset size of 1000 in each subsequent round, and without the re-initialization strategy

    python multiround_dpu.py --mode gcpu --without_reinit --updating_ratio 0.01 --model vgg --dataset cifar10 --dataset_size_init 1000 --dataset_size_subseq 1000   

(3) Random Partial Updating (RPU) with updating ratio 0.01, dataset size of 1000 before the initial deployment, dataset size of 1000 in each subsequent round, and without the re-initialization strategy    

    python multiround_rpu.py --mode rpu --without_reinit --updating_ratio 0.01 --model vgg --dataset cifar10 --dataset_size_init 1000 --dataset_size_subseq 1000   

(4) Full Updating (FU) with dataset size of 1000 before the initial deployment, dataset size of 1000 in each subsequent round, and with the same re-initialization in each round

    python multiround_fu.py --mode same_init --model vgg --dataset cifar10 --dataset_size_init 1000 --dataset_size_subseq 1000 


For more options, please refer to `python xxx.py -h` respectively.

### Results

The average accuracy difference over all rounds related to full updating (updating ratio k)

Method|MLP(k=0.005)|VGGNet(k=0.01)|ResNet56(k=0.1)
:---:|:---:|:---:|:---:
DPU|**-0.17%**|**+0.33%**|**-0.42%**
GCPU|-0.72%|-1.51%|-3.87%
RPU|-4.04%|-11.35%|-7.78%
Pruning|-1.45%|-4.35%|-2.35%


More results can be found in the paper.


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

