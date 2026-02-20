import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class PoseEncoder(nn.Module):
    def __init__(self):
        super(PoseEncoder, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        old_weights = resnet.conv1.weight.data
        new_weights = old_weights.repeat(1, 2, 1, 1) / 2.0

        self.encoder = resnet
        self.encoder.conv1 = nn.Conv2d(
            in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder.conv1.weight.data = new_weights

    
    def forward(self, I1, I2):
        # I1 = [B, 3, 192, 640]
        # I2 = [B, 3, 192, 640]
        x = torch.cat([I1, I2], dim=1) # [B, 6, 192, 640]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x) # [B, 512, 6, 20]

        return x


class PoseDecoder(nn.Module):
    def __init__(self, in_channel):
        super(PoseDecoder, self).__init__()
        self.layers = nn.ModuleDict({
            'pconv0' : nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1),
            'pconv1' : nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            'pconv2' : nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            'pconv3' : nn.Conv2d(in_channels=256, out_channels=6, kernel_size=1, stride=1)
        })
        self.ReLU = nn.ReLU()

    def forward(self, econv5):
        pconv0 = self.layers['pconv0'](econv5) # [B, 256, 6, 20]
        pconv0 = self.ReLU(pconv0) # [B, 256, 6, 20]

        pconv1 = self.layers['pconv1'](pconv0) # [B, 256, 6, 20]
        pconv1 = self.ReLU(pconv1) # [B, 256, 6, 20]

        pconv2 = self.layers['pconv2'](pconv1) # [B, 256, 6, 20]
        pconv2 = self.ReLU(pconv2) # [B, 256, 6, 20]

        pconv3 = self.layers['pconv3'](pconv2) # [B, 6, 6, 20]

        pose = pconv3.mean(dim=(2, 3)) # [B, 6]
        pose = 0.01 * pose

        return pose
    
class poseNetwork(nn.Module):
    def __init__(self):
        super(poseNetwork, self).__init__()
        self.encoder = PoseEncoder()
        self.decoder = PoseDecoder(512)

    def forward(self, I1, I2):
        out = self.encoder(I1, I2)
        out = self.decoder(out)

        return out