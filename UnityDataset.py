import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class UnityDataset(Dataset):
    def __init__(self, root_dirs):
        self.root_dir = root_dirs
        self.all_frames = []

        for root_dir in root_dirs:
            int_df = pd.read_csv(os.path.join(root_dir, 'intrinsics.csv'))

            f_ids = sorted(int_df['frame_id'].unique())

            for f_id in f_ids[:-2]:
                int_rows = int_df[int_df['frame_id'] == f_id].sort_values('camera_id').values
                
                self.all_frames.append({
                    'root': root_dir,
                    'frame_id': f_id,
                    'int_rows': int_rows
                })

    def __len__(self):
        return len(self.all_frames)
    
    def __getitem__(self, idx):
        data_info = self.all_frames[idx]
        root = data_info['root']
        f_id = data_info['frame_id']
        int_rows = data_info['int_rows']

        imgs = []

        for i in range(3):
            img_path = os.path.join(root, 'images', f'frame_{f_id+i:06d}_cam_{0}.jpg')
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).transpose(2, 0, 1)
            imgs.append(img)

        it = int_rows[0]
        k = [[it[2], 0,     it[4]],
            [0,     it[6], it[7]],
            [0,     0,     1]]

        return {
            'prev_image' : torch.from_numpy(imgs[0]).float() / 255.0,
            'target_image': torch.from_numpy(imgs[1]).float() / 255.0,
            'next_image': torch.from_numpy(imgs[2]).float() / 255.0,
            'intrinsics': torch.as_tensor(k, dtype=torch.float32),
        }