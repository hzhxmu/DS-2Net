import torch
import numpy as np
from thop import profile
from thop import clever_format
import math
from torch.optim.lr_scheduler import LambdaLR

# 裁剪梯度
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        
def cosine_lr_scheduler(optimizer, epoch, num_epochs, base_lr):
    t = (epoch / num_epochs)
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * t))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def one_cycle_lr(optimizer, epoch, total_epochs, max_lr, pct_start=0.3):
    if epoch < int(pct_start * total_epochs):
        lr = (max_lr / pct_start) * (epoch / total_epochs)
    else:
        lr = max_lr * (1.0 - (epoch - int(pct_start * total_epochs)) / (total_epochs * (1.0 - pct_start)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))