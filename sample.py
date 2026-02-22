import torch
import torch.nn as nn
import torch.nn.functional as F
from depthNetwork import depthNetwork
from poseNetwork import poseNetwork
from projector import projector

depthNetwork = depthNetwork()
poseNetwork = poseNetwork()
projector = projector()

B, C, H, W = 8, 3, 192, 640

dummy_image1 = torch.randn(B, 3, H, W)
dummy_image2 = torch.randn(B, 3, H, W)
dummy_intrinsic = torch.randn(B, 3, 3)
min_depth = 0.1
max_depth = 100.0
b = 1.0 / max_depth
a = 1.0 / min_depth - 1.0 / max_depth

disparity = depthNetwork(dummy_image1) # image1의 깊이
pose = poseNetwork(dummy_image1, dummy_image2) # image1에서 image2로의 회전과 이동

for i in range(4):
    disp = disparity[i]
    disp_resized = F.interpolate(disp, (H, W), mode='bilinear', align_corners=False)
    depth = 1.0 / (a * disp_resized + b)

    projected_image = projector(dummy_image2, depth, pose, dummy_intrinsic) # image2에 depth, pose, intrinsic를 적용해 나온 image

    print(projected_image.shape)