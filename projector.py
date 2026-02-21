import torch
import torch.nn as nn
import torch.nn.functional as F

class backprojectDepth(nn.Module):
    def __init__(self):
        super(backprojectDepth, self).__init__()

    def forward(self, depth_map, intrinsic):
        # depth_map : [B, 1, H, W]
        # intrinsic : [B, 3, 3]
        B, _, H, W = depth_map.shape
        grid = self.get_pixel_grid(depth_map) # [H, W, 2] [H, W, (x, y)]
        
        homogeneous_coordinates = self.get_homogeneous_coordinates(grid, intrinsic) # [B, H, W, 3]

        frustum = self.get_frustum(homogeneous_coordinates, depth_map)

        return frustum

    def get_pixel_grid(self, depth_map):
        B, _, H, W = depth_map.shape
        height = torch.arange(0, H, 1)
        width = torch.arange(0, W, 1)

        grid_y, grid_x = torch.meshgrid(height, width, indexing='ij') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=2) # [H, W, 2] [H, W, (x, y)]

        return grid
    
    def get_homogeneous_coordinates(self, grid, intrinsic):
        B = intrinsic.shape[0]
        H, W, _ = grid.shape

        ones = torch.ones((H, W, 1))
        grid = torch.cat([grid, ones], dim=2) # [H, W, 3] [H, W, (x, y, 1)]

        inv_intrinsic = torch.inverse(intrinsic) # [B, 3, 3]

        grid = grid.view(1, H, W, 1, 3)                     # [1, H, W, 1, 3]
        inv_intrinsic = inv_intrinsic.view(B, 1, 1, 3, 3)   # [B, 1, 1, 3, 3]

        homogeneous_coordinates = torch.matmul(inv_intrinsic, grid) # [B, H, W, 3, 1]
        return homogeneous_coordinates.squeeze(-1) # [B, H, W, 3]

    def get_frustum(self, homogeneous_coordinates, depth_map):
        B, H, W, _ = homogeneous_coordinates.shape

        depth_map = depth_map.permute(0, 2, 3, 1) # [B, H, W, 1]
        
        frustum = homogeneous_coordinates * depth_map # [B, H, W, 3]

        return frustum

class project3D(nn.Module):
    def __init__(self):
        super(project3D, self).__init__()

    def forward(self, frustum, pose):
        pass

class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()
        
    def forward(self, source_image, depth_map, pose, intrinsic):
        pass