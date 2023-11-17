# DSNet

## Requirements
- python 3.9
- pytorch 2.0.1
- torchvision 0.15.2

## Data Preparation
Downloading training and testing datasets and move them into ./dataset/, which can be found in this [Baidu Drive](https://pan.baidu.com/s/1OWuVwb6jAc27YHH7-IW7QQ ) [code: uf8a]
```
dataset
├── BUSI
│   ├── test
│   	├── images
│   	├── masks
│   ├── train
│   	├── images
│   	├── masks
├── DSB
│   ├── test
│   ├── train
├── Polyp
│   ├── test
│   	├── CVC-ClinicDB
│   	├── CVC-ColonDB
│   	├── ETIS-LaribPolypDB
│   	├── Kvasir
│   	├── test
│   ├── train
```

### Pretrained Backbone
You should download the pretrained backbone from [Baidu Drive](https://pan.baidu.com/s/1BBCUba8CN3oyxbbXyxNl5A) [code: 53dq], and then put it in the './pretrained_pth' folder for initialization.

## Training
Run the command scripts in `run/` to train models on different datasets. For example,  to train a breast ultrasound image segmentation model, run:
```
sh run/train_busi.sh
```

## Evaluating
You could download the trained model from and put the model in directory './model'.

