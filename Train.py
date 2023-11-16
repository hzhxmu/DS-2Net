import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.dsnetv2 import DSNetV2
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def softmax_with_temperature(arr, temperature=1.0):
    e_x = np.exp(arr / temperature)
    normalized_arr = e_x / e_x.sum()
    return normalized_arr

def test(model, path, dataset):
    if dataset == None:
        data_path = path
    else:
        data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res1, res2, res3, res4, res5, res6 = model(image)
        res = F.upsample(res1 + res2 + res3 + res4 + res5 + res6, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
    return DSC / num           

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = opt.size_rates
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3, P4, P5, P6 = model(images)
            
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss_P5 = structure_loss(P5, gts)
            loss_P6 = structure_loss(P6, gts)
            
            # ---- Adaptive Deep Supervision ----
            uncertanty1 = torch.abs(torch.sigmoid(P1) - 0.5)   
            uncertanty1 = torch.mean(1 - (uncertanty1 / 0.5)) 
            uncertanty2 = torch.abs(torch.sigmoid(P2) - 0.5)   
            uncertanty2 = torch.mean(1 - (uncertanty2 / 0.5))
            uncertanty3 = torch.abs(torch.sigmoid(P3) - 0.5)   
            uncertanty3 = torch.mean(1 - (uncertanty3 / 0.5))
            uncertanty4 = torch.abs(torch.sigmoid(P4) - 0.5)   
            uncertanty4 = torch.mean(1 - (uncertanty4 / 0.5))
            uncertanty5 = torch.abs(torch.sigmoid(P5) - 0.5)   
            uncertanty5 = torch.mean(1 - (uncertanty5 / 0.5))
            uncertanty6 = torch.abs(torch.sigmoid(P6) - 0.5)   
            uncertanty6 = torch.mean(1 - (uncertanty6 / 0.5))
            values = np.array([uncertanty1.cpu().detach().numpy(), uncertanty2.cpu().detach().numpy(), uncertanty3.cpu().detach().numpy(), 
                               uncertanty4.cpu().detach().numpy(), uncertanty5.cpu().detach().numpy(), uncertanty6.cpu().detach().numpy()])
            weights = softmax_with_temperature(values,1.0)                  
            weights = weights / max(weights)
            loss = weights[0] * loss_P1 + weights[1] * loss_P2 + weights[2] * loss_P3 + weights[3] * loss_P4 + weights[4] * loss_P5 + weights[5] * loss_P6
           
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    if (epoch + 1) % 1 == 0:
        if opt.dataset == "Polyp":
            for dataset_name in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                dataset_dice = test(model, opt.test_path, dataset_name)
                logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset_name, dataset_dice))
                print(dataset_name, ': ', dataset_dice)
        else:
            dataset_dice = test(model, opt.test_path, None)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, opt.dataset, dataset_dice))
            print(opt.dataset, ': ', dataset_dice)
        
        if opt.dataset == "Polyp":
            meandice = test(model, test_path, 'test')
        else:
            meandice = test(model, test_path, None)
            
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + '{}.pth'.format(opt.model_name + '_' + opt.dataset))
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer, AdamW, Adam and SGD')
    
    parser.add_argument('--dataset', type=str,
                        default=None, help='Polyp, BUSI and DSB')
    
    parser.add_argument('--size_rates',     
                        type=float,
                        nargs='+',
                        default=[0.75, 1, 1.25],
                        help='size_rates_list')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default=None,
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default=None,
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default=None)
    
    parser.add_argument('--train_log', type=str,
                        default=None)
    
    parser.add_argument('--model_name', type=str,
                        default=None)

    opt = parser.parse_args()
    
    if not os.path.exists(opt.train_log):
        os.makedirs(opt.train_log)
    
    logging.basicConfig(filename=opt.train_log + '{}.log'.format(opt.model_name + '_' + opt.dataset),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    model = DSNetV2().cuda()
    best = 0
    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
        
    print("tatal-best:{}".format(best))
    logging.info('~~~~~~~~tatal-best:{}~~~~~~~~'.format(best))
