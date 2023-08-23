import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class HandTeleopDataset(Dataset):

    def __init__(self, data):
        self.obs = data["obs"]
        self.next_obs = data["next_obs"]
        self.robot_qpos = data["robot_qpos"]
        self.actions = data["actions"]
        self.domains = data["domains"]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx])
        next_obs = torch.from_numpy(self.next_obs[idx])
        action = torch.from_numpy(self.actions[idx])
        robot_qpos = torch.from_numpy(self.robot_qpos[idx])
        domain = torch.tensor(self.domains[idx]).float()

        return {
            "obs": obs,
            "next_obs": next_obs,
            "robot_qpos": robot_qpos,
            "action": action,
            "domain": domain
        }


def collate_fn(samples):
    obs = torch.stack([x["obs"] for x in samples])
    next_obs = torch.stack([x["next_obs"] for x in samples])
    robot_qpos = torch.stack([x["robot_qpos"] for x in samples])
    action = torch.stack([x["action"] for x in samples])
    domain = torch.stack([x["domain"] for x in samples])

    return {
        "obs": obs,
        "next_obs": next_obs,
        "robot_qpos": robot_qpos,
        "action": action,
        "domain": domain
    }
