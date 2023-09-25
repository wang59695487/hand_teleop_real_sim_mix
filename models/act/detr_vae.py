# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
# from .backbone import build_backbone
# from .position_encoding import build_position_encoding
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, encoder, action_dim, num_queries,
            latent_dims, vision_dims, qpos_dims):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries

        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        #Robot qpos = 22 + 7
        self.input_proj_robot_state = nn.Linear(qpos_dims, hidden_dim)
        self.input_proj_obs = nn.Linear(vision_dims, hidden_dim)
        self.real_sim_bn = nn.BatchNorm1d(vision_dims)
      
        # encoder extra parameters
        self.latent_dim = latent_dims # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(qpos_dims, hidden_dim)  # project qpos to embedding
        self.encoder_obs_proj = nn.Linear(vision_dims, hidden_dim) # project obs to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.fea_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for features
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, obs, qpos, actions=None, is_pad=None):
        """
        obs: batch, features_dim
        qpos: batch, qpos_dim
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        obs = self.real_sim_bn(obs)
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            obs_embed = self.encoder_obs_proj(obs) # (bs, hidden_dim)
            obs_embed = torch.unsqueeze(obs_embed, axis=1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, obs_embed, action_embed], axis=1) # (bs, seq+3, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+3, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 3), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+3)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+3, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
      
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # Input feautes instead of image
        features = self.input_proj_obs(obs)
        hs = self.transformer(features, None, self.query_embed.weight, self.fea_pos_embed.weight, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(args):
    d_model = args.hidden_dims # 256
    dropout = args.dropout # 0.1
    nhead = args.n_heads # 8
    dim_feedforward = args.forward_dims # 2048
    num_encoder_layers = args.n_enc_layers # 4 # TODO shared with VAE decoder
    
    normalize_before = args.pre_norm # False
    
    activation = "relu"
    #activation = "gelu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_ACT_model(args):
    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        transformer,
        encoder,
        action_dim=args.action_dims,
        num_queries=args.n_queries,
        latent_dims=args.latent_dims,
        vision_dims=args.vision_dims,
        qpos_dims=args.qpos_dims
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
