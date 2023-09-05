import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HandTeleopDataset(Dataset):

    def __init__(self, demo_paths):
        self.demo_paths = demo_paths        

    def __len__(self):
        return len(self.demo_paths)

    def __getitem__(self, idx):
        with open(self.demo_paths[idx], "rb") as f:
            demo = pickle.load(f)

        return demo

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle, num_workers):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=lambda x: x)
