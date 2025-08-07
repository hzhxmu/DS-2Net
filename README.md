# DS^2Net
## DS^2Net: Detail-Semantic Deep Supervision Network for Medical Image Segmentation
[Paper Link](https://arxiv.org/abs/2508.04131)

## Requirements

- python 3.9
- pytorch 2.0.1
- torchvision 0.15.2

## Data Preparation

The datasets are provided in the [link](https://drive.google.com/drive/folders/16nMa5jvIFbU9xn5NxcwDfm6UVlVfQo2T).

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

The pretrained backbone is provided in the [link](https://drive.google.com/drive/folders/17rbCXDp1tNhwGwqRQgK6BNbDJ05Zewhd).

```
pretrained_pth
├── pvt_v2_b2.pth
```

## Training

Run the command scripts in `run/` to train models on different datasets. For example,  to train a breast ultrasound image segmentation model, run:

```
sh run/train_busi.sh
```

## Evaluating

The trained model are provided in the following [link](https://drive.google.com/drive/folders/1N1mqTv5YKJW0CchpYfqFgG9rxGhkdPcB). To evaluate them, download the model file and place it into `./model` and then run the command script in `run/`. For example,  to evaluate a breast ultrasound image segmentation model, run:

```
sh run/test_busi.sh
```

