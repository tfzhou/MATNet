## Motion-Attentive Transition for Zero-Shot Video Object Segmentation



This is a PyTorch implementation of our MATNet for unsupervised video object segmentation.

**Motion-Attentive Transition for Zero-Shot Video Object Segmentation.**

Tianfei Zhou, Shunzhou Wang, Yi Zhou, Yazhou Yao, Jianwu Li, Ling Shao, *AAAI 2020*, New York, USA.

## Prerequisites

The training and testing experiments are conducted using PyTorch 1.0.1 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
- [PyTorch 1.0.1](https://github.com/pytorch/pytorch)
                   
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

## Train

### Clone
```git clone --recursive https://github.com/tfzhou/MATNet.git```

### Download Datasets
In the paper, we use the following two public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-17](https://davischallenge.org/davis2017/code.html): we use all the data in the train subset of DAVIS-16. 
    However, please download DAVIS-17 to fit the code. It will automatically choose the subset of DAVIS-16 for training. 
- [YoutubeVOS-2018](https://youtube-vos.org/dataset/): we sample the training data every 10 frames in YoutubeVOS-2018.
- Create soft links:

    ```cd data; ln -s your/davis17/path DAVIS2017; ln -s your/youtubevos/path YouTubeVOS_2018;```
    
### Prepare Edge Annotations
I have provided some matlab scripts to generate edge annotations from mask. Please run ```data/run_davis2017.m``` 
and ```data/run_youtube.m```.

### Prepare HED Results
I have provided the pytorch codes to generate HED results for the two datasets (see ```3rdparty/pytorch-hed```).
Please run ```run_davis.py``` and ```run_youtube.py```. 

The codes are borrowed from https://github.com/sniklaus/pytorch-hed. 

### Prepare Optical Flow
I have provided the pytorch codes to generate optical flow results for the two datasets (see ```3rdparty/pytorch-pwc```).
Please run ```run_davis2017.py``` and ```run_youtubevos.py```. 

The codes are borrowed from https://github.com/sniklaus/pytorch-pwc. 
Please follow the [setup](https://github.com/sniklaus/pytorch-pwc#setup) section to install ```cupy```. 

`warning: Total size of optical flow results of Youtube-VOS is more than 30GB.`

### Train
Once all data is prepared, please run ```python train_MATNet.py``` for training.

## Test
1. Run ```python test_MATNet.py``` to obtain the saliency results on DAVIS-16 val set.
2. Run ```python apply_densecrf_davis.py``` for binary segmentation results.


## Segmentation Results

The segmentation results on DAVIS and Youtube-objects can be downloaded from [Google Drive](https://drive.google.com/file/d/1d23TGBtrr11g8KFAStwewTyxLq2nX4PT/view?usp=sharing).

## Pretrained Models

The pre-trained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1XlenYXgQjoThgRUbffCUEADS6kE4lvV_/view?usp=sharing).

## Citation
If you find MATNet useful for your research, please consider citing the following paper:
```
@inproceedings{zhou2020motion,
  title={Motion-Attentive Transition for Zero-Shot Video Object Segmentation},
  author={Tianfei Zhou, Shunzhou Wang, Yi Zhou, Yazhou Yao, Jianwu Li, Ling Shao},
  booktitle={Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020},
  organization={AAAI}
}
```


