import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from UnityDataset import UnityDataset
from depthNetwork import depthNetwork
from poseNetwork import poseNetwork
from projector import projector
from Loss import Minimum_Reprojection_Loss, Smooth_Loss
import os
import time

min_depth = 0.1
max_depth = 100.0
b = 1.0 / max_depth
a = 1.0 / min_depth - 1.0 / max_depth

batch_size = 8
H, W = 192, 640
learning_rate = 0.00005
Epoch = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dirs = [r'C:\Users\MSI\Desktop\DrivingData\Monodepth2\data_1']
full_dataset = UnityDataset(root_dirs)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

save_interval = 20

depth_model_save_path = r'./depth_model_save'
if not os.path.exists(depth_model_save_path): os.makedirs(depth_model_save_path)
pose_model_save_path = r'./pose_model_save'
if not os.path.exists(pose_model_save_path): os.makedirs(pose_model_save_path)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

depthNetwork = depthNetwork().to(device)
poseNetwork = poseNetwork().to(device)
projector = projector().to(device)

optimizer = optim.Adam(list(depthNetwork.parameters()) + list(poseNetwork.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

criterion_reprojection = Minimum_Reprojection_Loss().to(device)
criereion_smooth = Smooth_Loss().to(device)

def train():
    print('학습 시작')
    best_avg_test_loss = float('inf')
    for epoch in range(Epoch + 1):
        epoch_start_time = time.time()

        depthNetwork.train()
        poseNetwork.train()
        projector.train()

        train_loss = 0.0

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            prev_image = batch['prev_image'].to(device)
            target_image = batch['target_image'].to(device)
            next_image = batch['next_image'].to(device)
            intrins = batch['intrinsics'].to(device)
            
            optimizer.zero_grad()

            disparity = depthNetwork(target_image)

            pose1 = poseNetwork(target_image, prev_image)
            pose2 = poseNetwork(target_image, next_image)

            total_loss = 0

            for i in range(4):
                disp = disparity[i]
                disp_resized = F.interpolate(disp, (H, W), mode='bilinear', align_corners=False)
                depth = 1.0 / (a * disp_resized + b)
                
                projected_image_1 = projector(prev_image, depth, pose1, intrins)
                projected_image_2 = projector(next_image, depth, pose2, intrins)

                loss_p = criterion_reprojection(target_image, prev_image, next_image, projected_image_1, projected_image_2)
                loss_s = criereion_smooth(disp_resized, target_image)

                total_loss += (loss_p + 0.001 * loss_s)
            
            train_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{Epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()
            
        depthNetwork.eval()
        poseNetwork.eval()
        projector.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                prev_image = batch['prev_image'].to(device)
                target_image = batch['target_image'].to(device)
                next_image = batch['next_image'].to(device)
                intrins = batch['intrinsics'].to(device)

                disparity = depthNetwork(target_image)

                pose1 = poseNetwork(target_image, prev_image)
                pose2 = poseNetwork(target_image, next_image)

                total_loss = 0

                for i in range(4):
                    disp = disparity[i]
                    disp_resized = F.interpolate(disp, (H, W), mode='bilinear', align_corners=False)
                    depth = 1.0 / (a * disp_resized + b)

                    projected_image_1 = projector(prev_image, depth, pose1, intrins)
                    projected_image_2 = projector(next_image, depth, pose2, intrins)

                    loss_p = criterion_reprojection(target_image, prev_image, next_image, projected_image_1, projected_image_2)
                    loss_s = criereion_smooth(disp_resized, target_image)

                    total_loss += (loss_p + 0.001 * loss_s)

                test_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        epoch_end_time = time.time()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Test Loss : {avg_test_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % save_interval == 0:
            depth_save_path = os.path.join(depth_model_save_path, f'depth_model_epoch_{epoch}.pth')
            torch.save(depthNetwork.state_dict(), depth_save_path)
            pose_save_path = os.path.join(pose_model_save_path, f'pose_model_epoch_{epoch}.pth')
            torch.save(poseNetwork.state_dict(), pose_save_path)
            
            print(f'Saved : {depth_save_path}, {pose_save_path}')
        
        if avg_test_loss < best_avg_test_loss:
            best_avg_test_loss = avg_test_loss
            depth_save_path = os.path.join(depth_model_save_path, f'best_depth_model_epoch.pth')
            torch.save(depthNetwork.state_dict(), depth_save_path)
            pose_save_path = os.path.join(pose_model_save_path, f'best_pose_model_epoch.pth')
            torch.save(poseNetwork.state_dict(), pose_save_path)

            print(f'New Best Model Saved! Loss : {best_avg_test_loss:.4f}')    

        scheduler.step()    

if __name__ == "__main__":
    train(