import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCNetwork(nn.Module):
    def __init__(self, obs_dim, robot_qpos_dim, action_dim, hidden_dim=1200):
        super().__init__()

        self.l0 = nn.Linear(obs_dim, hidden_dim)
        self.real_bn0 = nn.BatchNorm1d(hidden_dim)
        self.sim_bn0 = nn.BatchNorm1d(hidden_dim)

        # self.l1 = nn.Linear(int(obs_dim/2), int(obs_dim/4))
        # self.real_bn1 = nn.BatchNorm1d(int(obs_dim/4))
        # self.sim_bn1 = nn.BatchNorm1d(int(obs_dim/4))

        # # self.l2 = nn.Linear(int(obs_dim/4), int(obs_dim/2))
        # # self.real_bn2 = nn.BatchNorm1d(int(obs_dim/2))
        # # self.sim_bn2 = nn.BatchNorm1d(int(obs_dim/2))

        self.vis_out = nn.Linear(hidden_dim, obs_dim)
        
        self.chin = obs_dim+robot_qpos_dim
        self.l3 = nn.Linear(self.chin, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.real_bn3 = nn.BatchNorm1d(hidden_dim)
        self.sim_bn3 = nn.BatchNorm1d(hidden_dim)

        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.real_bn4 = nn.BatchNorm1d(hidden_dim)
        self.sim_bn4 = nn.BatchNorm1d(hidden_dim)

        # self.l5 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn5 = nn.BatchNorm1d(hidden_dim)
        # self.real_bn5 = nn.BatchNorm1d(hidden_dim)
        # self.sim_bn5 = nn.BatchNorm1d(hidden_dim)

        # self.l6 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn6 = nn.BatchNorm1d(hidden_dim)
        # self.real_bn6 = nn.BatchNorm1d(hidden_dim)
        # self.sim_bn6 = nn.BatchNorm1d(hidden_dim)

        self.l7 = nn.Linear(hidden_dim, action_dim)

    
    def forward(self, concatenated_obs, robot_qpos, sim_real_label):
        
        if all(sim_real_label) and any(sim_real_label):
            x = torch.relu(self.real_bn0(self.l0(concatenated_obs)))
            # x = torch.relu(self.real_bn1(self.l1(x)))
            x = torch.cat([self.vis_out(x),robot_qpos], dim = 1)
            x = torch.relu(self.real_bn3(self.l3(x)))
            x = torch.relu(self.real_bn4(self.l4(x)))
            # x = torch.relu(self.real_bn5(self.l5(x)))
            # x = torch.relu(self.real_bn6(self.l6(x)))
            action = self.l7(x)

        elif not all(sim_real_label) and not any(sim_real_label):
           
            x = torch.relu(self.sim_bn0(self.l0(concatenated_obs)))
            # x = torch.relu(self.sim_bn1(self.l1(x)))
            x = torch.cat([self.vis_out(x),robot_qpos], dim = 1)
            x = torch.relu(self.sim_bn3(self.l3(x)))
            x = torch.relu(self.sim_bn4(self.l4(x)))
            # x = torch.relu(self.sim_bn5(self.l5(x)))
            # x = torch.relu(self.sim_bn6(self.l6(x)))
            action = self.l7(x)
            
        else:
            raise NotImplementedError
       
        return action


class InvFunction(nn.Module):
    """MLP for inverse dynamics model"""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state, next_state):
        joint_state = torch.cat([state, next_state], dim=1)
        return self.network(joint_state)