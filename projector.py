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

        frustum = self.get_frustum(homogeneous_coordinates, depth_map) # [B, H, W, 3]

        return frustum # [B, H, W, 3]

    def get_pixel_grid(self, depth_map):
        B, _, H, W = depth_map.shape
        height = torch.arange(0, H, 1, device=depth_map.device)
        width = torch.arange(0, W, 1, device=depth_map.device)

        grid_y, grid_x = torch.meshgrid(height, width, indexing='ij') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=2) # [H, W, 2] [H, W, (x, y)]

        return grid
    
    def get_homogeneous_coordinates(self, grid, intrinsic):
        B = intrinsic.shape[0]
        H, W, _ = grid.shape

        ones = torch.ones((H, W, 1), device=intrinsic.device)
        grid = torch.cat([grid, ones], dim=2) # [H, W, 3] [H, W, (x, y, 1)]

        inv_intrinsic = torch.inverse(intrinsic) # [B, 3, 3]

        grid = grid.view(1, H, W, 3, 1)                     # [1, H, W, 1, 3]
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
        # frustum : [B, H, W, 3]
        B, H, W, _ = frustum.shape
        ones = torch.ones((B, H, W, 1), device=frustum.device)
        frustum = torch.cat([frustum, ones], dim=3) # [B, H, W, 4]

        extrinsic = self.get_extrinsic(pose) # [B, 4, 4]

        frustum = frustum.view(B, H, W, 4, 1)
        extrinsic = extrinsic.view(B, 1, 1, 4, 4)

        new_points = torch.matmul(extrinsic, frustum).squeeze(-1) # [B, H, W, 4]

        return new_points[..., :3]

    def get_extrinsic(self, pose):
        B = pose.shape[0]
        axisangle = pose[:, :3] # [B, 3]
        translation = pose[:, 3:] # [B, 3]

        rotation = self.rot_from_axisangle(axisangle) # [B, 3, 3]
        under = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=pose.device) # [4]
        under = under.unsqueeze(0).unsqueeze(0).expand(B, 1, 4) #[B, 4]

        extrinsic = torch.cat([rotation, translation.unsqueeze(2)], dim=2) # [B, 3, 4]
        extrinsic = torch.cat([extrinsic, under], dim=1) # [B, 4, 4]

        return extrinsic

    def rot_from_axisangle(self, axisangle):
        # axisangle : [B, 3]
        theta = torch.norm(axisangle, p=2, dim=-1, keepdim=True)
        axis = axisangle / (theta + 1e-7)

        c = torch.cos(theta)
        s = torch.sin(theta)
        t = 1-c

        x = axis[:, 0:1] # [B, 1]
        y = axis[:, 1:2] # [B, 1]
        z = axis[:, 2:3] # [B, 1]

        row0 = torch.cat([t * x * x + c, t * x * y - s * z, t * z * x + s * y], dim=1) # [B, 3]
        row1 = torch.cat([t * x * y + s * z, t * y * y + c, t * y * z - s * x], dim=1) # [B, 3]
        row2 = torch.cat([t * z * x - s * y, t * y * z + s * x, t * z * z + c], dim=1) # [B, 3]

        rot = torch.stack([row0, row1, row2], dim=1) # [B, 3, 3]

        return rot

class project3Dto2D(nn.Module):
    def __init__(self):
        super(project3Dto2D, self).__init__() 

    def forward(self, points_3D, instrinsic, source_image):
        # points_3D : [B, H, W, 3]
        # instrinsic : [B, 3, 3]
        # source_image : [B, 3, H, W]
        B, H, W, _ = points_3D.shape

        points_3D = points_3D.view(B, H, W, 3, 1)
        instrinsic = instrinsic.view(B, 1, 1, 3, 3)

        coordinate = torch.matmul(instrinsic, points_3D).squeeze(-1) # [B, H, W, 3]

        x = coordinate[..., 0] / (coordinate[..., 2] + 1e-7) # [B, H, W]
        y = coordinate[..., 1] / (coordinate[..., 2] + 1e-7) # [B, H, W]

        grid_x = x / (W-1) * 2.0 - 1.0 # [B, H, W]
        grid_y = y / (H-1) * 2.0 - 1.0 # [B, H, W]

        grid_xy = torch.stack([grid_x, grid_y], dim=-1) # [B, H, W, 2]

        projected_image = F.grid_sample(source_image, grid_xy, padding_mode='zeros', align_corners=True) # [B, 3, H, W]

        return projected_image # [B, 3, H, W]

class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()
        self.backprojectDepth = backprojectDepth()
        self.project3D = project3D()
        self.project3Dto2D = project3Dto2D()
        
    def forward(self, source_image, depth_map, pose, intrinsic):
        # source_image : [B, 3, H, W]
        # depth_map : [B, 1, H, W]
        # pose : [B, 6]
        # intrinsic : [B, 3, 3]

        frustum = self.backprojectDepth(depth_map, intrinsic) # [B, H, W, 3]
        
        points_3D = self.project3D(frustum, pose) # [B, H, W, 3]

        projected_image = self.project3Dto2D(points_3D, intrinsic, source_image) # [B, 3, H, W]

        return projected_image # [B, 3, H, W]