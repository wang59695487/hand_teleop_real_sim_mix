# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

from models.act.detr_vae import build_ACT_model
# from dataset.act_dataset import set_seed

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False, exit_on_error=False)
    parser.add_argument("--lr", default=1e-4, type=float) # will be overridden
    parser.add_argument("--weight_decay", default=1e-2, type=float) # will be overridden
    parser.add_argument("--kl_weight", default=10, type=int)
    # Model parameters

    # * Transformer
    parser.add_argument("--enc_layers", default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward", default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--hidden_dim", default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument("--nheads", default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--num_queries", default=50, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument("--masks", action="store_true",
                        help="Train segmentation head if the flag is provided")
    
    ####Not used
    parser.add_argument("--real-demo-folder", default=None, type=str)
    parser.add_argument("--sim-demo-folder", default=None, type=str)
    parser.add_argument("--backbone-type", default="regnet_3_2gf")
    parser.add_argument("--eval-freq", default=100, type=int)
    parser.add_argument("--eval-start-epoch", default=400, type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--num-epochs", default=2000, type=int)
    parser.add_argument("--real-batch-size", default=32678, type=int)
    parser.add_argument("--sim-batch-size", default=32678, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--eval-randomness-scale", default=0, type=int)

  
    return parser

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def build_ACT_model_and_optimizer(args):
    # parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    # args = parser.parse_args()

    # for k, v in args_override.items():
    #     setattr(args, k, v)

    model = build_ACT_model(args)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters()]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.max_lr,
        weight_decay=args.wd_coef)

    return model, optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args.w_kl_loss
        print(f"KL Weight {self.kl_weight}")

    def forward(self, obs, qpos, actions=None, is_pad=None):
    
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(obs, qpos, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            return a_hat, mu, logvar
            # loss_dict = dict()
            # all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            # l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            # loss_dict["l1"] = l1
            # loss_dict["kl"] = total_kld[0]
            # loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            # return loss_dict

        else: # inference time
            a_hat, is_pad_hat, (mu, logvar) = self.model(obs, qpos) # no action, sample from prior
            return a_hat, mu, logvar

    def configure_optimizers(self):
        return self.optimizer

class ActAgent(object):
    def __init__(self, args):

        policy_config = {
            "lr": args.lr,
            "weight_decay": args.wd_coef,
            "num_queries": args.n_queries,
            "kl_weight": args.w_kl_loss,
            "hidden_dim": args.hidden_dims,
            "dim_feedforward": args.forward_dims,
            "enc_layers": args.n_enc_layers,
            "dec_layers": args.n_dec_layers,
            "nheads": args.n_heads,
        }

        # set_seed(args.seed)
        self.policy = ACTPolicy(policy_config)
        self.policy.to(args.device)
        self.optimizer = self.policy.configure_optimizers()

    def compute_loss(self, obs, qpos, actions, is_pad):
        loss_dict = self.policy(obs, qpos, actions, is_pad)
        return loss_dict
    
    def update_policy(self, loss):
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss.detach().cpu().item()
    
    def evaluate(self, obs, qpos, action=None, is_pad=None):
        pred_action = self.policy(obs, qpos)

        if action is not None:
            assert is_pad is not None
            all_l1 = F.l1_loss(pred_action, action, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            return l1.detach().cpu().item()
        
        return pred_action
    
    def save(self, weight_path, args):
        torch.save({
            "act_network_state_dict" : self.policy.state_dict(),
            "args" : args,
            }, weight_path
        )
    
    def load(self, weight_path):
        act_network_checkpoint = torch.load(weight_path)
        self.policy.load_state_dict(act_network_checkpoint["act_network_state_dict"])
        args = act_network_checkpoint["args"]       
        return args
    
        
        

