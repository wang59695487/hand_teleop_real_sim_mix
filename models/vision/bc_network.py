import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCNetwork(nn.Module):
    def __init__(self, obs_dim, robot_qpos_dim, action_dim, hidden_dim):
        super().__init__()

        # self.visual_network = nn.Sequential(
        #     #nn.LayerNorm(obs_dim),
        #     nn.Linear(obs_dim, int(obs_dim/2)),
        #     nn.GELU(),
        #     nn.Linear(int(obs_dim/2), int(obs_dim/4)),
        #     nn.Dropout(0.1)
        # )
        dropout = 0.1

        #self.chin = obs_dim+robot_qpos_dim

        self.l0 = nn.Linear(obs_dim, int(obs_dim/2))
        self.real_bn0 = nn.BatchNorm1d(int(obs_dim/2))
        self.sim_bn0 = nn.BatchNorm1d(int(obs_dim/2))

        self.l1 = nn.Linear(int(obs_dim/2), int(obs_dim/4))
        self.real_bn1 = nn.BatchNorm1d(int(obs_dim/4))
        self.sim_bn1 = nn.BatchNorm1d(int(obs_dim/4))

        # self.l2 = nn.Linear(int(obs_dim/4), int(obs_dim/2))
        # self.real_bn2 = nn.BatchNorm1d(int(obs_dim/2))
        # self.sim_bn2 = nn.BatchNorm1d(int(obs_dim/2))

        self.vis_out = nn.Linear(int(obs_dim/4), int(obs_dim/2))
        #self.vis_out = nn.Linear(int(obs_dim/4), int(obs_dim/2))
        self.vis_drop = nn.Dropout(dropout)
        
        self.chin = int(obs_dim/2)+robot_qpos_dim
        self.l3 = nn.Linear(self.chin, int(self.chin/2))
        self.bn3 = nn.BatchNorm1d(int(self.chin/2))
        self.real_bn3 = nn.BatchNorm1d(int(self.chin/2))
        self.sim_bn3 = nn.BatchNorm1d(int(self.chin/2))
        self.drop3 = nn.Dropout(dropout)

        self.l4 = nn.Linear(int(self.chin/2), int(self.chin/4))
        self.bn4 = nn.BatchNorm1d(int(self.chin/4))
        self.real_bn4 = nn.BatchNorm1d(int(self.chin/4))
        self.sim_bn4 = nn.BatchNorm1d(int(self.chin/4))
        self.drop4 = nn.Dropout(dropout)

        self.l5 = nn.Linear(int(self.chin/4), int(self.chin/2))
        self.bn5 = nn.BatchNorm1d(int(self.chin/2))
        self.real_bn5 = nn.BatchNorm1d(int(self.chin/2))
        self.sim_bn5 = nn.BatchNorm1d(int(self.chin/2))
        self.drop5 = nn.Dropout(dropout)

        # self.l6 = nn.Linear(int(self.chin/8), int(self.chin/4))
        # self.bn6 = nn.BatchNorm1d(int(self.chin/4))
        # self.real_bn6 = nn.BatchNorm1d(int(self.chin/4))
        # self.sim_bn6 = nn.BatchNorm1d(int(self.chin/4))

        self.l7 = nn.Linear(int(self.chin/2), action_dim)
        self.pn_drop = nn.Dropout(dropout)

    
    def forward(self, concatenated_obs, robot_qpos, sim_real_label):
        
        if all(sim_real_label) and any(sim_real_label):
            x = torch.relu(self.real_bn0(self.l0(concatenated_obs)))
            x = torch.relu(self.real_bn1(self.l1(x)))
            x = torch.cat([self.vis_drop(self.vis_out(x)),robot_qpos], dim = 1)
            x = torch.relu(self.real_bn3(self.l3(x)))
            x = torch.relu(self.real_bn4(self.l4(x)))
            x = torch.relu(self.real_bn5(self.l5(x)))
            # x = torch.relu(self.real_bn6(self.l6(x)))
            action = self.pn_drop(self.l7(x))

        elif not all(sim_real_label) and not any(sim_real_label):
           
            x = torch.relu(self.sim_bn0(self.l0(concatenated_obs)))
            x = torch.relu(self.sim_bn1(self.l1(x)))
            x = torch.cat([self.vis_drop(self.vis_out(x)),robot_qpos], dim = 1)
            x = torch.relu(self.sim_bn3(self.l3(x)))
            x = torch.relu(self.sim_bn4(self.l4(x)))
            x = torch.relu(self.sim_bn5(self.l5(x)))
            # x = torch.relu(self.sim_bn6(self.l6(x)))
            action = self.pn_drop(self.l7(x))
            
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