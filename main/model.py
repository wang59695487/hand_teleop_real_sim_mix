from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from policy.act_agent import ACTPolicy


class ClassToken(nn.Module):

    def __init__(self, cls_token_idx=0):
        super().__init__()
        self.cls_token_idx = cls_token_idx

    def forward(self, tokens):
        if self.cls_token_idx == 0:
            img_feats = tokens["img_feats"][:, self.cls_token_idx]
            return img_feats
        else:
            return tokens["img_feats"].squeeze(dim=-1).squeeze(dim=-1)


class MLPLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )


class MLP(nn.Sequential):

    def __init__(self, in_channels, hidden_channels, out_channels=None):
        n_channels = [in_channels] + hidden_channels
        layers = []
        for in_ch, out_ch in zip(n_channels[:-1], n_channels[1:]):
            layers.append(MLPLayer(in_ch, out_ch))

        if out_channels is not None:
            layers.append(nn.Linear(hidden_channels[-1], out_channels))
        super().__init__(*layers)


class Agent(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # self.init_vision_net(args)
        self.init_policy_net(args)

    def init_vision_net(self, args):
        model_fn = eval(f"models.{args.backbone}")
        model = model_fn(weights="IMAGENET1K_V1")
        self.vision_net = nn.Sequential(
            create_feature_extractor(model, {"avgpool": "img_feats"}),
            ClassToken("img_feats")
        )

    def init_policy_net(self, args):
        self.policy_net = ACTPolicy(args)

    # def forward(self, images, robot_qpos, obj_pose=None,
    #         action=None, is_pad=None):
    #     vis_feats = self.get_image_feats(images)
    #     if obj_pose is not None:
    #         # add current object pose
    #         robot_qpos = torch.cat([robot_qpos, obj_pose], dim=-1)
    #     actions_pred, mu, log_var = self.policy_net(vis_feats, robot_qpos,
    #         action, is_pad)

    #     return actions_pred, mu, log_var

    def forward(self, images, robot_qpos, action=None, is_pad=None):
        if self.args.dann:
            actions_pred, mu, log_var, domain_preds = self.policy_net(images,
                robot_qpos, action, is_pad)

            return actions_pred, mu, log_var, domain_preds
        else:
            actions_pred, mu, log_var = self.policy_net(images, robot_qpos,
                action, is_pad)

            return actions_pred, mu, log_var

    @torch.no_grad()
    def get_action(self, images, qpos, ret_tensor=True):
        action = self(images, qpos)[0]
        if not ret_tensor:
            action = action.cpu().numpy()

        return action

    def get_image_feats(self, images):
        if self.vision_net.training:
            feats = self.vision_net(images)
        else:
            with torch.no_grad():
                feats = self.vision_net(images)

        return feats


if __name__ == "__main__":
    transformer = nn.Transformer(batch_first=True)
    inputs = torch.rand(4, 1, 512)
    targets = torch.zeros(4, 1, 512)
    outputs = transformer(inputs, targets)
    print(outputs)
