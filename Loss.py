import torch
import torch.nn as nn
import torch.nn.functional as F

def SSIM(x, y):
    # x, y : [B, 3, H, W]

    # C1과 C2는 안정제 역할
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    x = F.pad(x, (1, 1, 1, 1), mode='reflect')
    y = F.pad(y, (1, 1, 1, 1), mode='reflect')

    # mu는 픽셀의 평균 밝기
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    # sigma는 픽셀의 분산, 낮으면 주변이랑 비슷, 높으면 주변이랑 매우다름
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y # 이건 공분산

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return SSIM_n / SSIM_d # [B, 3, H, W]

class photometric_error(nn.Module):
    def __init__(self):
        super(photometric_error, self).__init__()
        self.a = 0.85

    def forward(self, image_A, image_B):
        ssim = SSIM(image_A, image_B) # [B, 3, H, W]
        ssim = torch.clamp((1 - ssim) / 2, 0, 1).mean(1, keepdim=True) # [B, 1, H, W]
        l1 = torch.abs(image_A - image_B).mean(1, keepdim=True) # [B, 1, H, W]

        pe = self.a * ssim + (1 - self.a) * l1

        return pe        

class Minimum_Reprojection_Loss(nn.Module):
    def __init__(self):
        super(Minimum_Reprojection_Loss, self).__init__()
        self.pe = photometric_error()

    def forward(self, target_image, source_image1, source_image2, projected_image_1, projected_image_2):
        # target_image : [B, 3, H, W]
        # projected_image : [B, 3, H, W]

        projected_pe1 = self.pe(target_image, projected_image_1) # [B, 1, H, W]
        projected_pe2 = self.pe(target_image, projected_image_2) # [B, 1, H, W]
        min_projected_pe = torch.cat([projected_pe1, projected_pe2], dim=1) # [B, 2, H, W]
        min_projected_pe = torch.min(min_projected_pe, dim=1, keepdim=True)[0] # [B, 1, H, W]

        source_pe1 = self.pe(target_image, source_image1) # [B, 1, H, W]
        source_pe2 = self.pe(target_image, source_image2) # [B, 1, H, W]
        min_source_pe = torch.cat([source_pe1, source_pe2], dim=1) # [B, 2, H, W]
        min_source_pe = torch.min(min_source_pe, dim=1, keepdim=True)[0] # [B, 1, H, W]

        mask = (min_projected_pe < min_source_pe).float() # [B, 1, H, W]

        loss = min_projected_pe * mask # [B, 1, H, W]
        
        return loss.mean() # [B, 1]
    
class Smooth_Loss(nn.Module):
    def __init__(self):
        super(Smooth_Loss, self).__init__()

    def forward(self, disp, image):
        # disp : [B, 1, H, W]
        # image : [B, 3, H, W]

        # disp의 기울기 (x, y 방향)
        mean_disp = disp.mean(2, keepdim=True).mean(3, keepdim=True)
        disp = disp / (mean_disp + 1e-7)

        disp_dx = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        disp_dy = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # image의 기울기 (x, y 방향)
        image_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(1, keepdim=True)
        image_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(1, keepdim=True)

        weights_x = torch.exp(-image_dx * 10.0)
        weights_y = torch.exp(-image_dy * 10.0)

        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        
        return smoothness_x.mean() + smoothness_y.mean()