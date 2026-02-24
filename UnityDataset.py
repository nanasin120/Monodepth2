import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random
import os

class UnityDataset(Dataset):
    def __init__(self, root_dirs, augment=False):
        self.root_dir = root_dirs
        self.augment = augment
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
        self.brightness=0.2
        self.contrast=0.2
        self.saturation=0.2
        self.hue=0.1

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.all_frames)
    
    def __getitem__(self, idx):
        data_info = self.all_frames[idx]
        root = data_info['root']
        f_id = data_info['frame_id']
        int_rows = data_info['int_rows']

        imgs = []
        it = int_rows[0]
        k = np.array([[it[2], 0,     it[4]],
            [0,     it[6], it[7]],
            [0,     0,     1]], dtype=np.float32)

        for i in range(3):
            img_path = os.path.join(root, 'images', f'frame_{f_id+i:06d}_cam_{0}.jpg')
            img = Image.open(img_path).convert('RGB')
            imgs.append(img)

        if self.augment:
            b_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            # Contrast: [max(0, 1-0.2), 1+0.2]
            c_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            # Saturation: [max(0, 1-0.2), 1+0.2]
            s_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            # Hue: [-0.1, 0.1]
            h_factor = random.uniform(-self.hue, self.hue)
            transforms_list = [
                lambda x: TF.adjust_brightness(x, b_factor),
                lambda x: TF.adjust_contrast(x, c_factor),
                lambda x: TF.adjust_saturation(x, s_factor),
                lambda x: TF.adjust_hue(x, h_factor)
            ]
            random.shuffle(transforms_list)
            for t in transforms_list:
                imgs = [t(img) for img in imgs]

            if random.random() > 0.5:
                imgs = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in imgs]

                width = imgs[0].width
                k[0, 2] = width - k[0, 2]


        prev_data = self.to_tensor(imgs[0])
        target_data = self.to_tensor(imgs[1])
        next_data = self.to_tensor(imgs[2])

        return {
            'prev_image' : prev_data,
            'target_image': target_data,
            'next_image': next_data,
            'intrinsics': torch.from_numpy(k)
        }