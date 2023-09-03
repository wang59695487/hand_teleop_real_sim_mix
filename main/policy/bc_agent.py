import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision.encoder import *
from models.vision.bc_network import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BCSSAgent(object):
    def __init__(self, backbone_type, embedding_name, concatenated_obs_shape,robot_qpos_shape, state_shape, action_shape, hidden_dim, 
                 hidden_state_dim, state_encoder_output_dim, num_layers, num_filters, in_channels,
                 num_shared_layers, num_frozen_layers, train_policy, train_visual_encoder, train_state_encoder,
                 train_inv, use_visual_encoder, use_visual_backbone, use_ss, use_state_encoder, ss_visual_encoder_lr, ss_state_encoder_lr,
                 ss_inv_lr, bc_lr, bc_beta, weight_decay, frame_stack=4):
        self.frame_stack_num = frame_stack
        self.weight_decay = weight_decay
        self.bc_visual_encoder = None
        self.bc_state_encoder = None

        if use_visual_encoder:
            assert concatenated_obs_shape == None
            self.bc_visual_encoder, visual_encoder_out_dim = make_encoder(use_visual_backbone, backbone_type, embedding_name, in_channels, num_layers, num_filters)
            self.bc_visual_encoder = self.bc_visual_encoder.to(device)
        
        if use_state_encoder:
            assert state_shape != None
            assert hidden_state_dim != None
            assert state_encoder_output_dim != None
            self.bc_state_encoder = StateEncoder(state_shape, hidden_state_dim, state_encoder_output_dim).to(device)
        
        if concatenated_obs_shape is not None:
            assert state_shape == None
        elif use_state_encoder:
            concatenated_obs_shape = self.frame_stack_num*(visual_encoder_out_dim)+state_encoder_output_dim
        else:
            concatenated_obs_shape = self.frame_stack_num*(visual_encoder_out_dim)+state_shape

        print("Concatenated Observation Shape for Policy Network: {}".format(concatenated_obs_shape))
        self.bc_policy_network = BCNetwork(concatenated_obs_shape, robot_qpos_shape, action_shape, hidden_dim).to(device)    

        self.num_frozen_layers = num_frozen_layers
        # self-supervision
        self.inv = None
        self.ss_visual_encoder = None
        self.ss_state_encoder = None

        if use_ss:
            assert use_visual_encoder == True
            self.ss_visual_encoder,_ = make_encoder(use_visual_backbone, backbone_type, embedding_name, 
                                                  in_channels, num_layers, num_filters)
            self.ss_visual_encoder = self.ss_visual_encoder.to(device)

            self.ss_visual_encoder.copy_conv_weights_from(self.bc_visual_encoder, num_shared_layers)
            if use_state_encoder:
                self.ss_state_encoder = StateEncoder(state_shape, hidden_state_dim, state_encoder_output_dim).to(device)
                self.ss_state_encoder.copy_fc_weights_from(self.bc_state_encoder, n=1) # n=1 or n=2

            self.inv = InvFunction(concatenated_obs_shape, action_shape, hidden_dim).to(device)
            self.ss_update_freq = 100

        self.train(train_visual_encoder, train_state_encoder, train_policy, train_inv)
        
        self.init_bc_optimizers(bc_lr, bc_beta, train_state_encoder, train_policy, train_visual_encoder)
        if use_ss:
            self.init_ss_optimizers(ss_visual_encoder_lr, ss_inv_lr, ss_state_encoder_lr)

    def init_bc_optimizers(self, bc_lr, bc_beta, train_state_encoder, train_policy, train_visual_encoder):
        params = [p for p in self.bc_policy_network.parameters()]
        if train_state_encoder and self.bc_state_encoder!=None:
            state_encoder_params = [p for p in self.bc_state_encoder.parameters() if p.requires_grad]
            params = state_encoder_params + params
        if train_visual_encoder and self.bc_visual_encoder!=None:
            visual_encoder_params = [p for p in self.bc_visual_encoder.parameters() if p.requires_grad]
            params = visual_encoder_params + params
        #self.bc_module_optimizer = torch.optim.AdamW(params=params, lr=bc_lr, weight_decay=0.01)
        self.bc_module_optimizer = torch.optim.AdamW(params=params, lr=bc_lr, betas=[0.9,0.999], weight_decay=self.weight_decay, eps=1.0e-8)

    def init_ss_optimizers(self, ss_visual_encoder_lr, inv_lr, ss_state_encoder_lr):
        params = [p for p in self.ss_visual_encoder.parameters() if p.requires_grad]
        self.visual_encoder_optimizer =  torch.optim.Adam(params, lr=ss_visual_encoder_lr)        
        if self.ss_state_encoder != None:
            params = [p for p in self.ss_state_encoder.parameters() if p.requires_grad]
            self.state_encoder_optimizer = torch.optim.Adam(params, lr=ss_state_encoder_lr)
        self.inv_optimizer =  torch.optim.Adam(self.inv.parameters(), lr=inv_lr)        

    def train(self, train_visual_encoder, train_state_encoder, train_policy, train_inv):
        if self.bc_visual_encoder is not None:
            if not train_visual_encoder:
                self.bc_visual_encoder.train(train_visual_encoder)
            else:    
                if type(self.bc_visual_encoder) == ResNetEncoder:
                    if self.num_frozen_layers == -1 or self.num_frozen_layers == None:
                        self.bc_visual_encoder.train(train_visual_encoder)
                    else:
                        ct = 0
                        for child in self.bc_visual_encoder.model.children():
                            ct += 1
                            if ct < self.num_frozen_layers:
                                for param in child.parameters():
                                    param.requires_grad = False
                else:
                    self.bc_visual_encoder.train(train_visual_encoder)
        
        if self.bc_state_encoder is not None:
            self.bc_state_encoder.train(train_state_encoder)
        
        self.bc_policy_network.train(train_policy)

        if self.inv is not None:
            assert self.ss_visual_encoder is not None
            if not train_visual_encoder:
                self.ss_visual_encoder.train(train_visual_encoder)
            else:
                if type(self.ss_visual_encoder) == ResNetEncoder:
                    if self.num_frozen_layers == -1 or self.num_frozen_layers == None:    
                        self.ss_visual_encoder.train(train_visual_encoder)
                    else:          
                        ct = 0
                        for child in self.ss_visual_encoder.model.children():
                            ct += 1
                            if ct < self.num_frozen_layers:
                                for param in child.parameters():
                                    param.requires_grad = False   
                else:             
                    self.ss_visual_encoder.train(train_visual_encoder)

            if self.ss_state_encoder is not None:
                self.ss_state_encoder.train(train_state_encoder)

            self.inv.train(train_inv)

    def update_inv(self, h, s, next_h, next_s, action, L=None, step=None):
        '''
        h: stacked visual obs batch
        s: stacked robot states batch
        h_next: stacked next visual obs batch
        s_next: stacked next robot states batch
        '''
        batch_size = h.shape[0]
        frame_stack = int(h.shape[1]/3)
        embedding_batch = []
        for frame_id in range(frame_stack):
            if isinstance(self.ss_visual_encoder, EmbeddingNet):
                processed_obs = h[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)
            else:
                processed_obs = h[:, frame_id*3:(frame_id+1)*3]
            embedding_i = self.ss_visual_encoder(processed_obs)
            embedding_batch.append(embedding_i)
        embedding_batch = torch.stack(embedding_batch, dim=1)
        embedding_batch = embedding_batch.view(batch_size, -1) # concat frames
        
        embedding_next_batch = []
        for frame_id in range(frame_stack):
            if isinstance(self.ss_visual_encoder, EmbeddingNet):
                processed_obs = next_h[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)
            else:
                processed_obs = next_h[:, frame_id*3:(frame_id+1)*3]
            embedding_i = self.ss_visual_encoder(processed_obs)
            embedding_next_batch.append(embedding_i)
        embedding_next_batch = torch.stack(embedding_next_batch, dim=1)
        embedding_next_batch = embedding_next_batch.view(batch_size, -1) # concat frames

        if self.ss_state_encoder is not None:
            state_batch = self.ss_state_encoder(s)
            state_next_batch = self.ss_state_encoder(next_s)
        else:
            state_batch = s
            state_next_batch = next_s
        
        concatenated_obs = torch.cat((embedding_batch, state_batch), dim=1)
        concatenated_next_obs = torch.cat((embedding_next_batch, state_next_batch), dim=1)
        pred_action = self.inv(concatenated_obs, concatenated_next_obs)
        inv_loss = F.mse_loss(pred_action, action)

        self.visual_encoder_optimizer.zero_grad()
        if self.ss_state_encoder is not None:
            self.state_encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()
        inv_loss.backward()

        self.visual_encoder_optimizer.step()
        if self.ss_state_encoder is not None:
            self.state_encoder_optimizer.step()        
        self.inv_optimizer.step()

        if L is not None:
            L.log('train/inv_loss', inv_loss.item(), step)

        return inv_loss.item()

    def compute_loss(self, obs=None, state=None, next_obs=None, next_state=None, action=None, robot_qpos=None, L=None, step=None, 
                        concatenated_obs=None, concatenated_next_obs=None, sim_real_label=None):
        if concatenated_obs is None:
            assert obs != None
            batch_size = obs.shape[0]
            frame_stack = int(obs.shape[1]/3)
            assert self.frame_stack_num == frame_stack
            assert concatenated_next_obs == None
            if self.bc_visual_encoder is not None:
                embedding_batch = []
                for frame_id in range(frame_stack):
                    if isinstance(self.bc_visual_encoder, EmbeddingNet):
                        processed_obs = obs[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)
                    else:
                        processed_obs = obs[:, frame_id*3:(frame_id+1)*3]
                    embedding_i = self.bc_visual_encoder(processed_obs) # obs: bxfsx256x256 -> bx256x256xfs
                    embedding_batch.append(embedding_i)
                embedding_batch = torch.stack(embedding_batch, dim=1)
                embedding_batch = embedding_batch.view(batch_size, -1) # concat frames
            if self.bc_state_encoder is not None:
                state_batch = self.bc_state_encoder(state)
            else:
                state_batch = state
            concatenated_obs = torch.cat((embedding_batch, state_batch), dim=1)
            # print(embedding_batch.shape, state_batch.shape, concatenated_obs.shape)
        else:
            #print("##############################inv network is not using##############################")
            assert self.inv == None

        pred_action = self.bc_policy_network(concatenated_obs, robot_qpos, sim_real_label)
        action = action.type(torch.float32)

        # if L is not None:
        #     L.log('train/bc_loss_total', bc_loss.item(), step)
        
        # self.bc_module_optimizer.zero_grad()
        # bc_loss.backward()
        # self.bc_module_optimizer.step()

        return F.mse_loss(pred_action, action)

    
    def update(self, bc_loss, L=None, step=None):
               
        if L is not None:
            L.log('train/bc_loss_total', bc_loss.item(), step)
        
        self.bc_module_optimizer.zero_grad()
        bc_loss.backward()
        self.bc_module_optimizer.step()

        # if self.inv is not None and step % self.ss_update_freq == 0:
        #     next_obs = next_obs.to(device)
        #     next_state = next_state.to(device)
        #     self.update_inv(obs, state, next_obs, next_state, action, L, step)     

        return bc_loss.detach().cpu().item()

    def validate(self, obs=None, state=None, action=None, robot_qpos=None, L=None, step=None, concatenated_obs=None,sim_real_label=None, mode='eval'):
        with torch.no_grad():
            if concatenated_obs is None:
                frame_stack = int(obs.shape[1]/3)
                assert self.frame_stack_num == frame_stack
                if self.bc_visual_encoder is not None:
                    valid_embedding = []
                    for frame_id in range(frame_stack):
                        if isinstance(self.bc_visual_encoder, EmbeddingNet):
                            processed_obs = obs[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)
                        else:
                            processed_obs = obs[:, frame_id*3:(frame_id+1)*3]
                        embedding_i = self.bc_visual_encoder(processed_obs) # obs: bxfsx256x256 -> bx256x256xfs
                        if self.bc_visual_encoder.training:
                            embedding_i = embedding_i.cpu().numpy()
                        valid_embedding.append(torch.from_numpy(embedding_i).to(device))
                    valid_embedding = torch.stack(valid_embedding, dim=1)
                    valid_embedding = valid_embedding.view(len(obs), -1) # concat frames
                if self.bc_state_encoder is not None:
                    state_batch = self.bc_state_encoder(state)
                else:
                    state_batch = state
                concatenated_obs = torch.cat((valid_embedding, state_batch), dim=1)
            else:
                assert self.inv == None
            
            pred_action = self.bc_policy_network(concatenated_obs,robot_qpos, sim_real_label)
            if mode=='eval':
                valid_loss = F.mse_loss(pred_action, action)
                if L is not None:
                    L.log('eval/loss', valid_loss.item()/len(action), step)
                
                return valid_loss.detach().cpu().item()

            elif mode=='test':
                return pred_action            

    def save(self, weight_path, args):
        torch.save({
            'policy_network_state_dict' : self.bc_policy_network.state_dict(),
            'args' : args,
            }, weight_path
        )

    # def save(self, model_dir, args):
    #     model_dir = "./trained_models/{}".format(model_dir)
    #     os.makedirs(model_dir, exist_ok=True)
    #     torch.save({
    #                 'policy_network_state_dict' : self.bc_policy_network.state_dict(),
    #                 'args' : args,
    #                 }, '%s/bc_model.pt' % (model_dir)
    #     )
    #     if self.bc_visual_encoder is not None:
    #         torch.save(self.bc_visual_encoder.state_dict(), '%s/bc_visual_encoder.pt'%(model_dir))
    #     if self.bc_state_encoder is not None:
    #         torch.save(self.bc_state_encoder.state_dict(), '%s/bc_state_encoder.pt'%(model_dir))    
    #     if self.inv is not None:
    #         torch.save(
    #             self.inv.state_dict(),
    #             '%s/inv.pt' % (model_dir)
    #         )
    #     if self.ss_visual_encoder is not None:
    #         torch.save(
    #             self.ss_visual_encoder.state_dict(),
    #             '%s/ss_visual_encoder.pt' % (model_dir)
    #         )
    #     if self.ss_state_encoder is not None:
    #         torch.save(
    #             self.ss_state_encoder.state_dict(),
    #             '%s/ss_state_encoder.pt' % (model_dir)
    #         )

    def load(self, weight_path):
        policy_network_checkpoint = torch.load(weight_path)
        self.bc_policy_network.load_state_dict(policy_network_checkpoint['policy_network_state_dict'])
        args = policy_network_checkpoint['args']       
        return args


def make_agent(concatenated_obs_shape, action_shape, robot_qpos_shape, state_shape, args, frame_stack=4):
    '''
    if concatenated_obs is used, user has to specify its shape and state_shape should be none.
    frame stacking can only be 4 for now.
    '''
    return BCSSAgent(
        backbone_type=args['backbone_type'],
        embedding_name=args['embedding_name'],
        concatenated_obs_shape=concatenated_obs_shape,
        robot_qpos_shape=robot_qpos_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_dim=args['hidden_dim'],
        hidden_state_dim=args['state_hidden_dim'],
        state_encoder_output_dim=args['state_encoder_output_dim'],
        num_layers=args['num_layers'],
        num_filters=args['num_filters'],
        in_channels=args['in_channels'],
        num_shared_layers=args['num_shared_layers'],
        num_frozen_layers=args['num_frozen_layers'],
        use_visual_encoder=args['use_visual_encoder'],
        use_visual_backbone=args['use_visual_backbone'],
        use_ss=args['use_ss'],
        use_state_encoder=args['use_state_encoder'],
        train_policy=args['train_policy'],
        train_visual_encoder=args['train_visual_encoder'],
        train_state_encoder=args['train_state_encoder'],
        train_inv=args['train_inv'],
        ss_visual_encoder_lr=args['ss_visual_encoder_lr'],
        ss_state_encoder_lr=args['ss_state_encoder_lr'],
        ss_inv_lr=args['ss_inv_lr'],
        bc_lr=args['bc_lr'],
        bc_beta=args['bc_beta'],
        weight_decay=args['weight_decay'],        
        frame_stack=frame_stack
    )