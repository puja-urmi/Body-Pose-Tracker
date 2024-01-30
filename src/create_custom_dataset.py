#!/usr/bin/env python
# coding: utf-8

# In[21]:


from torch.utils.data import Dataset
from skimage import io
import numpy as np
import pandas as pd
import torch
import os


# In[22]:


class create_custom_dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.kp_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.kp_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.kp_frame.iloc[idx, 0])
        image = io.imread(img_name)
        kp = self.kp_frame.iloc[idx, 1:]
        kp = np.array([kp], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'kp': kp}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[ ]:




