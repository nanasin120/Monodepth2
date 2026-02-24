import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
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
learning_rate = 0.0001
Epoch = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_interval = 25

def train():
    print('학습 시작')
    best_avg_test_loss = float('inf')
    for epoch in range(Epoch + 1):
        epoch_start_time = time.time()

        depth_model.train()
        pose_model.train()
        project_model.train()

        train_loss = 0.0

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            prev_image = batch['prev_image'].to(device)
            target_image = batch['target_image'].to(device)
            next_image = batch['next_image'].to(device)
            intrins = batch['intrinsics'].to(device)
            
            optimizer.zero_grad()

            disparity = depth_model(target_image)

            pose1 = pose_model(target_image, prev_image)
            pose2 = pose_model(target_image, next_image)

            total_loss = 0

            for i in range(4):
                disp = disparity[i]
                disp_resized = F.interpolate(disp, (H, W), mode='bilinear', align_corners=False)
                depth = 1.0 / (a * disp_resized + b)
                
                projected_image_1 = project_model(prev_image, depth, pose1, intrins)
                projected_image_2 = project_model(next_image, depth, pose2, intrins)

                loss_p = criterion_reprojection(target_image, prev_image, next_image, projected_image_1, projected_image_2)
                loss_s = criterion_smooth(disp_resized, target_image)

                total_loss += (loss_p + 0.001 * loss_s) / 4
            
            train_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{Epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()
            
        depth_model.eval()
        pose_model.eval()
        project_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                prev_image = batch['prev_image'].to(device)
                target_image = batch['target_image'].to(device)
                next_image = batch['next_image'].to(device)
                intrins = batch['intrinsics'].to(device)

                disparity = depth_model(target_image)

                pose1 = pose_model(target_image, prev_image)
                pose2 = pose_model(target_image, next_image)

                total_loss = 0

                for i in range(4):
                    disp = disparity[i]
                    disp_resized = F.interpolate(disp, (H, W), mode='bilinear', align_corners=False)
                    depth = 1.0 / (a * disp_resized + b)

                    projected_image_1 = project_model(prev_image, depth, pose1, intrins)
                    projected_image_2 = project_model(next_image, depth, pose2, intrins)

                    loss_p = criterion_reprojection(target_image, prev_image, next_image, projected_image_1, projected_image_2)
                    loss_s = criterion_smooth(disp_resized, target_image)

                    total_loss += (loss_p + 0.001 * loss_s) / 4

                test_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        epoch_end_time = time.time()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Test Loss : {avg_test_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % save_interval == 0:
            depth_save_path = os.path.join(depth_model_save_path, f'depth_model_epoch_{epoch+version}.pth')
            torch.save(depth_model.state_dict(), depth_save_path)
            pose_save_path = os.path.join(pose_model_save_path, f'pose_model_epoch_{epoch+version}.pth')
            torch.save(pose_model.state_dict(), pose_save_path)
            
            print(f'Saved : {depth_save_path}, {pose_save_path}')
        
        if avg_test_loss < best_avg_test_loss:
            best_avg_test_loss = avg_test_loss
            depth_save_path = os.path.join(depth_model_save_path, f'best_depth_model_epoch.pth')
            torch.save(depth_model.state_dict(), depth_save_path)
            pose_save_path = os.path.join(pose_model_save_path, f'best_pose_model_epoch.pth')
            torch.save(pose_model.state_dict(), pose_save_path)

            print(f'New Best Model Saved! Loss : {best_avg_test_loss:.4f}')    

        scheduler.step()    

if __name__ == "__main__":
    depth_model_save_path = r'./depth_model_save'
    if not os.path.exists(depth_model_save_path): os.makedirs(depth_model_save_path)
    pose_model_save_path = r'./pose_model_save'
    if not os.path.exists(pose_model_save_path): os.makedirs(pose_model_save_path)

    root_dirs = [r'C:\Users\MSI\Desktop\DrivingData\Monodepth2\data_1']
    full_dataset = UnityDataset(root_dirs)
    dataset_size = len(full_dataset)

    torch.manual_seed(1004) # 다시 해도 데이터가 똑같게

    indices = torch.randperm(dataset_size).tolist()
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    test_indices = indices[split:]

    print(f"Total: {dataset_size}, Train: {len(train_indices)}, Test: {len(test_indices)}")

    train_dataset_raw = UnityDataset(root_dirs,augment=True)
    test_dataset_raw = UnityDataset(root_dirs,augment=False)

    train_dataset = Subset(train_dataset_raw, train_indices)
    test_dataset = Subset(test_dataset_raw, test_indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      # CPU 코어 수에 맞춰 조절 (보통 4~8)
        pin_memory=True,    # CUDA 사용 시 필수
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    depth_model = depthNetwork().to(device)
    pose_model = poseNetwork().to(device)
    project_model = projector().to(device)
    version = 0

    if version == -1 :
        depth_model.load_state_dict(torch.load(os.path.join(depth_model_save_path, 'best_depth_model_epoch.pth'), map_location=device))
        pose_model.load_state_dict(torch.load(os.path.join(pose_model_save_path, 'best_pose_model_epoch.pth'), map_location=device))
        print("Loaded Best Model")
    elif version > 0 :
        depth_path = os.path.join(depth_model_save_path, f'depth_model_epoch_{version}.pth')
        pose_path = os.path.join(pose_model_save_path, f'pose_model_epoch_{version}.pth')
        
        if os.path.exists(depth_path) and os.path.exists(pose_path):
            depth_model.load_state_dict(torch.load(depth_path, map_location=device))
            pose_model.load_state_dict(torch.load(pose_path, map_location=device))
            print(f"Loaded Version {version} Model")
        else:
            print(f"Warning: Model version {version} not found! Starting from scratch.")
            version = 0

    optimizer = optim.Adam(list(depth_model.parameters()) + list(pose_model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250], gamma=0.1)

    criterion_reprojection = Minimum_Reprojection_Loss().to(device)
    criterion_smooth = Smooth_Loss().to(device)
    
    train()