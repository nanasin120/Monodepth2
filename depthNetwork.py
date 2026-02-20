import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x): 
        # x : [B, 3, 192, 640]
        econv1 = self.conv1(x) # [B, 64, 96, 320]
        econv1 = self.bn1(econv1) # [B, 64, 96, 320]
        econv1 = self.relu(econv1) # [B, 64, 96, 320]

        econv2 = self.layer1(self.maxpool(econv1)) # [B, 64, 48, 160]
        econv3 = self.layer2(econv2) # [B, 128, 24, 80]
        econv4 = self.layer3(econv3) # [B, 256, 12, 40]
        econv5 = self.layer4(econv4) # [B, 512, 6, 20]

        return [econv5, econv4, econv3, econv2, econv1]

class DepthDecoder(nn.Module):
    def __init__(self, encoder_channel): # encoder_channel = [64, 64, 128, 256, 512]
        super(DepthDecoder, self).__init__()
        self.layers = nn.ModuleDict({
            'upconv5': nn.Conv2d(in_channels=encoder_channel[4], out_channels=256, kernel_size=3, stride=1, padding=1),
            'iconv5' : nn.Conv2d(in_channels=256 + encoder_channel[3], out_channels=256, kernel_size=3, stride=1, padding=1),

            'upconv4': nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            'iconv4' : nn.Conv2d(in_channels=128 + encoder_channel[2], out_channels=128, kernel_size=3, stride=1, padding=1),
            'disp4'  : nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),

            'upconv3': nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            'iconv3' : nn.Conv2d(in_channels=64 + encoder_channel[1], out_channels=64, kernel_size=3, stride=1, padding=1),
            'disp3'  : nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),

            'upconv2': nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            'iconv2' : nn.Conv2d(in_channels=32 + encoder_channel[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            'disp2'  : nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),

            'upconv1': nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            'iconv1' : nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            'disp1'  : nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        })
        self.ELU = nn.ELU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, encoder_features):
        econv5, econv4, econv3, econv2, econv1 = encoder_features

        # --- Stage 5 ---
        # econv5 : [B, 512, 6, 20]
        # econv4 : [B, 256, 12, 40]
        upconv5 = self.layers['upconv5'](econv5) # [B, 256, 6, 20]
        upconv5 = self.ELU(upconv5) # [B, 256, 6, 20]
        upconv5 = F.interpolate(upconv5, scale_factor=2, mode='nearest') # [B, 256, 12, 40]

        iconv5 = torch.cat([upconv5, econv4], dim=1) # [B, 512, 12, 40]
        iconv5 = self.layers['iconv5'](iconv5) # [B, 256, 12, 40]
        iconv5 = self.ELU(iconv5) # [B, 256, 12, 40]

        # --- Stage 4 ---
        # econv3 : [B, 128, 24, 80]
        upconv4 = self.layers['upconv4'](iconv5) # [B, 128, 12, 40]
        upconv4 = self.ELU(upconv4) # [B, 128, 12, 40]
        upconv4 = F.interpolate(upconv4, scale_factor=2, mode='nearest') # [B, 128, 24, 80]

        iconv4 = torch.cat([upconv4, econv3], dim=1) # [B, 256, 24, 80]
        iconv4 = self.layers['iconv4'](iconv4) # [B, 128, 24, 80]
        iconv4 = self.ELU(iconv4) # [B, 128, 24, 80]

        disp4 = self.layers['disp4'](iconv4) # [B, 1, 24, 80]
        disp4 = self.Sigmoid(disp4) # [B, 1, 24, 80]

        # --- Stage 3 ---
        # econv2 : [B, 64, 48, 160]
        upconv3 = self.layers['upconv3'](iconv4) # [B, 64, 24, 80]
        upconv3 = self.ELU(upconv3) # [B, 64, 24, 80]
        upconv3 = F.interpolate(upconv3, scale_factor=2, mode='nearest') # [B, 64, 48, 160]

        iconv3 = torch.cat([upconv3, econv2], dim=1) # [B, 128, 48, 160]
        iconv3 = self.layers['iconv3'](iconv3) # [B, 64, 48, 160]
        iconv3 = self.ELU(iconv3) # [B, 64, 48, 160]

        disp3 = self.layers['disp3'](iconv3) # [B, 1, 48, 160]
        disp3 = self.Sigmoid(disp3) # [B, 1, 48, 160]

        # --- Stage 2 ---
        # econv1 : [B, 64, 96, 320]
        upconv2 = self.layers['upconv2'](iconv3) # [B, 32, 48, 160]
        upconv2 = self.ELU(upconv2) # [B, 32, 48, 160]
        upconv2 = F.interpolate(upconv2, scale_factor=2, mode='nearest') # [B, 32, 96, 320]

        iconv2 = torch.cat([upconv2, econv1], dim=1) # [B, 64, 96, 320]
        iconv2 = self.layers['iconv2'](iconv2) # [B, 32, 96, 320]
        iconv2 = self.ELU(iconv2) # [B, 32, 96, 320]

        disp2 = self.layers['disp2'](iconv2) # [B, 1, 96, 320]
        disp2 = self.Sigmoid(disp2) # [B, 1, 96, 320]

        # --- Stage 1 ---
        upconv1 = self.layers['upconv1'](iconv2) # [B, 16, 96, 320]
        upconv1 = self.ELU(upconv1) # [B, 16, 96, 320]
        upconv1 = F.interpolate(upconv1, scale_factor=2, mode='nearest') # [B, 16, 192, 640]

        iconv1 = self.layers['iconv1'](upconv1) # [B, 16, 192, 640]
        iconv1 = self.ELU(iconv1) # [B, 16, 192, 640]

        disp1 = self.layers['disp1'](iconv1) # [B, 1, 192, 640]
        disp1 = self.Sigmoid(disp1) # [B, 1, 192, 640]

        return [disp1, disp2, disp3, disp4]

class depthNetwork(nn.Module):
    def __init__(self):
        super(depthNetwork, self).__init__()
        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder(encoder_channel=[64, 64, 128, 256, 512])

    def forward(self, x):
        encoder_features = self.encoder(x)
        disparity = self.decoder(encoder_features)

        return disparity 