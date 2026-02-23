import torch
import torch.nn as nn
import torch.nn.functional as F
from UnityDataset import UnityDataset
from depthNetwork import depthNetwork
from poseNetwork import poseNetwork
from projector import projector
import random
import matplotlib
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

depthNetwork = depthNetwork().to(device)
poseNetwork = poseNetwork().to(device)
projector = projector().to(device)

depthNetwork.load_state_dict(torch.load(r'depth_model_save\best_depth_model_epoch.pth', weights_only=True))
poseNetwork.load_state_dict(torch.load(r'pose_model_save\best_pose_model_epoch.pth', weights_only=True))

root_dirs = [r'C:\Users\MSI\Desktop\DrivingData\Monodepth2\data_1']
full_dataset = UnityDataset(root_dirs)

sample_idx = random.randint(0, len(full_dataset))
sample = full_dataset[sample_idx]

prev_image = sample['prev_image'].unsqueeze(0).to(device)
target_image = sample['target_image'].unsqueeze(0).to(device)
next_image = sample['next_image'].unsqueeze(0).to(device)
intrins = sample['intrinsics'].unsqueeze(0).to(device)

depthNetwork.eval()

with torch.no_grad():   
    disparity = depthNetwork(target_image)
    disp = disparity[0]

target_image = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
target_image = (target_image * 255).astype(np.uint8)

pil_img = Image.fromarray(target_image)
pil_img.show()

disp = disp.squeeze(0).squeeze(0).cpu().numpy()
disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-7)
#disp = (disp * 255).astype(np.uint8)
magma_map = matplotlib.colormaps['magma']
disp_magma = magma_map(disp) # [H, W, 4]

disp_magma = (disp_magma[:, :, :3] * 255).astype(np.uint8)

pil_img = Image.fromarray(disp_magma)
pil_img.show()