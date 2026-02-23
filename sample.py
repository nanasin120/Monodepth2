import torch
import torch.nn as nn
import torch.nn.functional as F
from depthNetwork import depthNetwork
from poseNetwork import poseNetwork
from projector import projector
from Loss import Minimum_Reprojection_Loss, Smooth_Loss

depthNetwork = depthNetwork()
poseNetwork = poseNetwork()
projector = projector()

B, C, H, W = 8, 3, 192, 640

dummy_target_image = torch.randn(B, 3, H, W)
dummy_source_image1 = torch.randn(B, 3, H, W)
dummy_source_image2 = torch.randn(B, 3, H, W)
dummy_intrinsic = torch.randn(B, 3, 3)

min_depth = 0.1
max_depth = 100.0
b = 1.0 / max_depth
a = 1.0 / min_depth - 1.0 / max_depth

disparity = depthNetwork(dummy_target_image) # image1의 깊이
pose1 = poseNetwork(dummy_target_image, dummy_source_image1) # image1에서 image2로의 회전과 이동
pose2 = poseNetwork(dummy_target_image, dummy_source_image2) # image1에서 image2로의 회전과 이동

criterion_reprojection = Minimum_Reprojection_Loss()
criereion_smooth = Smooth_Loss()

total_loss = 0

for i in range(4):
    disp = disparity[i]
    disp_resized = F.interpolate(disp, (H, W), mode='bilinear', align_corners=False)
    depth = 1.0 / (a * disp_resized + b)

    projected_image_1 = projector(dummy_source_image1, depth, pose1, dummy_intrinsic) # image2에 depth, pose, intrinsic를 적용해 나온 image
    projected_image_2 = projector(dummy_source_image2, depth, pose2, dummy_intrinsic) # image2에 depth, pose, intrinsic를 적용해 나온 image

    loss_p = criterion_reprojection(dummy_target_image, dummy_source_image1, dummy_source_image2, projected_image_1, projected_image_2)
    loss_s = criereion_smooth(disp_resized, dummy_target_image)

    total_loss += (loss_p + 0.001 * loss_s)