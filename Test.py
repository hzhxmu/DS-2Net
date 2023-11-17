import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from tqdm import tqdm
from lib.dsnetv2 import DSNetV2
from utils.dataloader import test_dataset
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def test(data_path, save_path, data_name):
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    num = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    print("-------------Generate Result Map-------------")
    for i in tqdm(range(num)):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        P1, P2, P3, P4, P5, P6 = model(image)
        res = F.upsample(P1 + P2 + P3 + P4 + P5 + P6, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
        
    print(data_name, 'Finish!')  
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Polyp", help='Polyp, BUSI and DSB')
    parser.add_argument('--test_path', type=str, default="./dataset/Polyp/test/", help='test image')   
    parser.add_argument('--model_pth', type=str, default="./model/DSNet_Polyp.pth", help='model_pth')
    parser.add_argument('--result_map', type=str, default="./result_map/", help='result_map_save_pth')
    parser.add_argument('--model_name', type=str, default="DSNet")
    parser.add_argument('--testsize', type=int, default=352)
    opt = parser.parse_args()
    
    if opt.dataset == "Polyp":
        data = ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        
    model = DSNetV2()
    model.load_state_dict(torch.load(opt.model_pth))
    model.cuda()
    model.eval()
    
    if opt.dataset == "Polyp":
        for data_name in data:
            data_path = opt.test_path + '{}'.format(data_name)
            save_path = opt.result_map + '{}/{}/'.format(opt.model_name + "_" + opt.dataset, data_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            test(data_path, save_path, data_name)
    else:
        data_path = opt.test_path
        save_path = opt.result_map + '{}/'.format(opt.model_name + "_" + opt.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        test(data_path, save_path, opt.dataset)
