import torch
import torch.nn as nn
import torch.nn.functional as F
from depthNetwork import depthNetwork
from poseNetwork import poseNetwork

poseNetwork = poseNetwork()

B, C, H, W = 8, 3, 192, 640

dummy_input_1 = torch.randn(B, C, H, W)
dummy_input_2 = torch.randn(B, C, H, W)

pose = poseNetwork(dummy_input_1, dummy_input_2)

print(pose.shape)