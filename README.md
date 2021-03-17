# 面向目标检测的对抗样本
## Introduce

    目标检测作为对抗样本领域中的新对象，本仓库选择这方面较为经典的方法进行分析比较，代码涉及到的方法有：DAG、RAP、UEA、TOG四种方法.

## Requirements

* python == 3.6.6
* torch == 1.2.0 
* cupy-cuda100
* opencv == 4.4.0.44
* scipy == 1.2.1

## Usage
    
    整个项目有四个文件夹，其中方法对应的是DAG、RAP、UEA、TOG，其中测试用的Faster Rcnn代码见simple-faster-rcnn文件。

##Acknowledgemen
    该项目是在下列仓库的基础下进一步改进开发的：

* [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
* [git-disl/TOG](https://github.com/git-disl/TOG)
* [LiangSiyuan21/Adversarial-Attacks-for-Image-and-Video-Object-Detection](https://github.com/LiangSiyuan21/Adversarial-Attacks-for-Image-and-Video-Object-Detection/tree/master/img_attack_with_attention)
* [yuezunli/BMVC2018R-AP](https://github.com/yuezunli/BMVC2018R-AP#requirements)
* [cihangxie/DAG](https://github.com/cihangxie/DAG)

