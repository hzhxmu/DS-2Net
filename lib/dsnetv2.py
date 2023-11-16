import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbone import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3
import os
import torchvision.models as models
import math

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.dropout(self.conv1(x))
        return self.sigmoid(x)
    
class DEM(nn.Module):
    def __init__(self, channel, HR_Kernel_Size, HR_Padding, add = True):
        super(DEM, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        self.add = add
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(1, 1, kernel_size = HR_Kernel_Size, padding = HR_Padding, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size = HR_Kernel_Size, padding = HR_Padding, bias=False)

    def forward(self, high_feature, low_feature):
        high_feature = self.upsample(high_feature)
        fusion_feature = torch.cat([high_feature, low_feature], dim=1)
        feature_pool = self.conv_pool(fusion_feature)
        x, _ = torch.max(low_feature, dim=1, keepdim=True)
        x = self.conv2(self.conv1(x))
        mask = torch.sigmoid(x)
        boundly_enhance = feature_pool * mask

        if self.add:
            return boundly_enhance + low_feature
        else:
            return boundly_enhance

class SEM(nn.Module):
    def __init__(self, channel):
        super(SEM, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channel)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, high_feature, low_feature):
        high_feature = self.upsample(high_feature)
        attention_map = self.sa(high_feature)
        attention_map = self.conv1(attention_map)
        attention_map = torch.sigmoid(attention_map)
        x1 = low_feature * attention_map
        x1 = self.ca(x1) * x1
        return x1 + high_feature   

class DSNetV2(nn.Module):
    def __init__(self, channel=32):
        super(DSNetV2, self).__init__()
        self.backbone1 = pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone1.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone1.load_state_dict(model_dict)
        
        self.Translayer1_0 = Conv2D(64, channel, 1, padding = 0, act = False)
        self.Translayer1_1 = Conv2D(128, channel, 1, padding = 0, act=False)
        self.Translayer1_2 = Conv2D(320, channel, 1, padding = 0, act=False)
        self.Translayer1_3 = Conv2D(512, channel, 1, padding = 0, act=False)
        
        self.DEM1 = DEM(channel, HR_Kernel_Size=3, HR_Padding=1, add = True)
        self.DEM2 = DEM(channel, HR_Kernel_Size=5, HR_Padding=2, add = True)
        self.DEM3 = DEM(channel, HR_Kernel_Size=7, HR_Padding=3, add = False)
        
        self.SEM1 = SEM(channel)
        self.SEM2 = SEM(channel)
        self.SEM3 = SEM(channel)
        
        self.output_conv1 = nn.Conv2d(channel, 1, 1)
        self.output_conv2 = nn.Conv2d(channel, 1, 1)
        self.output_conv3 = nn.Conv2d(channel, 1, 1)
        self.output_conv4 = nn.Conv2d(channel, 1, 1)
        self.output_conv5 = nn.Conv2d(channel, 1, 1)
        self.output_conv6 = nn.Conv2d(channel, 1, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        pvt = self.backbone1(x)
        pvt1 = pvt[0]    #(1,64,88,88)
        pvt2 = pvt[1]    #(1,128,44,44)
        pvt3 = pvt[2]    #(1,320,22,22)
        pvt4 = pvt[3]    #(1,512,11,11)
        
        pvt1 = self.Translayer1_0(pvt1)    #(1,32,88,88)  
        pvt2 = self.Translayer1_1(pvt2)    #(1,32,44,44) 
        pvt3 = self.Translayer1_2(pvt3)    #(1,32,22,22) 
        pvt4 = self.Translayer1_3(pvt4)    #(1,32,11,11) 
    
        x1_1 = self.DEM1(pvt4, pvt3)    #(1,32,22,22) 
        x1_2 = self.DEM2(x1_1, pvt2)    #(1,32,44,44) 
        x1_3 = self.DEM3(x1_2, pvt1)    #(1,32,88,88) 
        
        x2_1 = self.SEM1(pvt4, pvt3)    #(1,32,22,22) 
        x2_2 = self.SEM2(x2_1, pvt2)    #(1,32,44,44)
        x2_3 = self.SEM3(x2_2, pvt1)    #(1,32,88,88) 
        
        output1 = self.output_conv1(x1_1)
        output2 = self.output_conv2(x1_2)
        output3 = self.output_conv3(x1_3)
        output4 = self.output_conv4(x2_1)
        output5 = self.output_conv5(x2_2)
        output6 = self.output_conv6(x2_3)
        
        prediction1 = F.interpolate(output1, scale_factor=16, mode='bilinear')
        prediction2 = F.interpolate(output2, scale_factor=8, mode='bilinear') 
        prediction3 = F.interpolate(output3, scale_factor=4, mode='bilinear') 
        prediction4 = F.interpolate(output4, scale_factor=16, mode='bilinear')
        prediction5 = F.interpolate(output5, scale_factor=8, mode='bilinear') 
        prediction6 = F.interpolate(output6, scale_factor=4, mode='bilinear') 
        return prediction1, prediction2, prediction3, prediction4, prediction5, prediction6

if __name__ == '__main__':
    model = DSNetV2().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    from thop import profile
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"Number of parameters: {params/1e6}M")
    print(f"Computational complexity (FLOPs): {flops/1e9}G")
