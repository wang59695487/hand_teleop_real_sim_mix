import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb
from omegaconf import OmegaConf

import dataset.bc_dataset as bc_dataset
from main.policy.bc_agent import make_agent
#from att_agent import make_agent

from eval import apply_IK_get_real_action
from feature_extractor import generate_feature_extraction_model
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from logger import Logger
from tqdm import tqdm
from dataset.bc_dataset import RandomShiftsAug, argument_dependecy_checker, prepare_real_sim_data
from hand_teleop.real_world import lab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(agent, validation_loader, L, epoch):
    loss_val = 0
    for iter, data_batch in enumerate(validation_loader):
        valid_obs = data_batch[0].to(device)
        valid_actions = data_batch[-2].to(device)
        sim_real_label = data_batch[-1].to(device)
        if len(data_batch) == 6:
            valid_states = data_batch[2].to(device)
            loss = agent.validate(obs=valid_obs, state=valid_states, action=valid_actions, L=L, step=epoch, mode='eval') 
        else:
            valid_robot_qpos = data_batch[2].to(device)
            valid_states = None
            loss = agent.validate(concatenated_obs=valid_obs, action=valid_actions, robot_qpos=valid_robot_qpos, L=L, step=epoch, sim_real_label=sim_real_label, mode='eval')
        loss_val += loss

    loss_val /= len(validation_loader)

    return loss_val

def compute_loss(agent, it_per_epoch, bc_train_dataloader, bc_validation_dataloader, L, epoch):
    for _ in tqdm(range(it_per_epoch)):
        data_batch = next(iter(bc_train_dataloader))
        obs_batch = data_batch[0].to(device)
        next_obs_batch = data_batch[1].to(device)
        action_batch = data_batch[-2].to(device)
        sim_real_label = data_batch[-1].to(device)

        if len(data_batch) == 6:
            state_batch = data_batch[2].to(device)
            next_state_batch = data_batch[3].to(device)
        else:
            state_batch = None
            next_state_batch = None
            robot_qpos_batch = data_batch[2].to(device)

        if state_batch is not None:
            loss = agent.compute_loss(obs=obs_batch, state=state_batch, next_obs=next_obs_batch, next_state=next_state_batch,  action=action_batch, L=L, step=epoch, concatenated_obs=None, concatenated_next_obs=None,sim_real_label=sim_real_label)
        else:
            loss = agent.compute_loss(concatenated_obs=obs_batch, concatenated_next_obs=next_obs_batch, action=action_batch, robot_qpos=robot_qpos_batch, sim_real_label=sim_real_label, L=L, step=epoch)

    return loss

def main(args):
    # read and prepare data 
    Prepared_Data = prepare_real_sim_data(args['dataset_folder'],args["backbone_type"],args['batch_size'],args['batch_size'],
                                 args['val_ratio'], seed = 20230806)
    it_per_epoch = Prepared_Data['it_per_epoch']
    bc_train_set = Prepared_Data['bc_train_set']
    bc_train_dataloader = Prepared_Data['bc_train_dataloader']
    bc_validation_dataloader = Prepared_Data['bc_validation_dataloader']
    print('Data prepared')
    if 'state' in bc_train_set.dummy_data.keys():
        state_shape = len(bc_train_set.dummy_data['state'])
        obs_shape = bc_train_set.dummy_data['obs'].shape
        concatenated_obs_shape = None
        print("State Shape: {}".format(state_shape))
        print("Observation Shape: {}".format(obs_shape))
    else:
        state_shape = None
        concatenated_obs_shape = len(bc_train_set.dummy_data['obs'])
        print("Concatenated Observation (State + Visual Obs) Shape: {}".format(concatenated_obs_shape))
    action_shape = len(bc_train_set.dummy_data['action'])
    robot_qpos_shape = len(bc_train_set.dummy_data['robot_qpos'])
    print("Action shape: {}".format(action_shape))
    print("robot_qpos shape: {}".format(robot_qpos_shape))
    # make agent
    agent = make_agent(
                       concatenated_obs_shape=concatenated_obs_shape, 
                       action_shape=action_shape, 
                       state_shape=state_shape, 
                       robot_qpos_shape=robot_qpos_shape,
                       args=args, 
                       frame_stack=args['frame_stack']
                       )
    L = Logger("{}_{}".format(args['model_name'],args['num_epochs']))

    if args['use_augmentation']:
        aug = RandomShiftsAug()
        aug = aug.to(device)

    it = 0
    # wandb.log({
    #     'dataset_folder': args['dataset_folder'],
    #     'batch_size': args['batch_size'],
    #     'backbone_type' : args['backbone_type'],
    #     'ss_visual_encoder_lr': args['ss_visual_encoder_lr'],
    #     'ss_state_encoder_lr': args['ss_state_encoder_lr'],
    #     'ss_inv_lr': args['ss_inv_lr'],
    #     'bc_lr': args['bc_lr'],
    #     'bc_beta': args['bc_beta']})
    
    if not args["eval_only"]:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", f"{args['task']}_{args['backbone_type']}_{cur_time}")
        #wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
        #wandb.login(key=wandb_cfg.key)
        wandb.init(
            project="hand-teleop",
            name=os.path.basename(log_dir),
            config=args
        )
        os.makedirs(log_dir, exist_ok=True)
        best_success = 0
        for epoch in range(args['num_epochs']):
            print('  ','Epoch: ', epoch)
            loss_train = 0
            for _ in tqdm(range(it_per_epoch)):
                it += 1
                time_train_iter = time.time()
                data_batch = next(iter(bc_train_dataloader))
                obs_batch = data_batch[0].to(device)
                next_obs_batch = data_batch[1].to(device)
                action_batch = data_batch[-2].to(device)
                sim_real_label = data_batch[-1].to(device)
        
                if len(data_batch) == 6:
                    state_batch = data_batch[2].to(device)
                    next_state_batch = data_batch[3].to(device)
                else:
                    state_batch = None
                    next_state_batch = None
                    robot_qpos_batch = data_batch[2].to(device)

                if args['use_augmentation'] and len(obs_batch.shape)==4:
                    obs_batch = aug(obs_batch)

                if state_batch is not None:
                    loss = agent.update(obs=obs_batch, state=state_batch, next_obs=next_obs_batch, next_state=next_state_batch,  action=action_batch, L=L, step=epoch, concatenated_obs=None, concatenated_next_obs=None,sim_real_label=sim_real_label)
                else:
                    loss = agent.update(concatenated_obs=obs_batch, concatenated_next_obs=next_obs_batch, action=action_batch, robot_qpos=robot_qpos_batch, sim_real_label=sim_real_label, L=L, step=epoch)
                loss_train += loss
            loss_train /= it_per_epoch

            agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)

            loss_val = evaluate(agent, bc_validation_dataloader, L, epoch)

            metrics = {
                "loss/train": loss_train,
                "loss/val": loss_val,
                "epoch": epoch
            }

            if (epoch + 1) % args["eval_freq"] == 0 and (epoch+1) >= args["eval_start_epoch"]:
                #total_steps = x_steps * y_steps = 4 * 5 = 20
                #avg_success = eval_in_env(args, agent, log_dir, epoch + 1, 4, 5)
                #metrics["avg_success"] = avg_success
                agent.save(os.path.join(log_dir, f"epoch_{epoch + 1}.pt"), args)
                if avg_success > best_success:
                    agent.save(os.path.join(log_dir, f"epoch_best.pt"), args)

            agent.train(train_visual_encoder=args['train_visual_encoder'],
                        train_state_encoder=args['train_state_encoder'], 
                        train_policy=args['train_policy'], 
                        train_inv=args['train_inv'])

            wandb.log(metrics)

        agent.load(os.path.join(log_dir, "epoch_best.pt"))
        agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)
        #final_success = eval_in_env(args, agent, log_dir, "best", 10, 10)

        # wandb.log({"final_success": final_success})
        # print(f"Final success rate: {final_success:.4f}")

        wandb.finish()
    else:
        log_dir = os.path.dirname(args["ckpt"])
        wandb.init(
            project="hand-teleop",
            name=log_dir+args["ckpt"],
            config=args
        )
        os.makedirs(log_dir, exist_ok=True)
        agent.load(args["ckpt"])
        agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)
        final_success = eval_in_env(args, agent, log_dir, "best", 10, 10)
        wandb.log({"final_success": final_success})
        print(f"Final success rate: {final_success:.4f}")

        wandb.finish()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--demo-folder", required=True)
    parser.add_argument("--backbone-type", default="resnet50")
    parser.add_argument("--eval-freq", default=200, type=int)
    parser.add_argument("--eval-start-epoch", default=400, type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default=None, type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    args = {
        'dataset_folder': args.demo_folder,
        'batch_size': 65536,
        'val_ratio': 0.1,
        'bc_lr': 2e-5,
        'num_epochs': 1600,
        'weight_decay': 1e-2,
        'model_name': '',
        'resume': False,
        'load_model_from': None,
        'use_augmentation': False,
        'save': True,
        'save_freq': 100,
        'use_visual_encoder': False,
        'use_visual_backbone': False,
        'backbone_type' : args.backbone_type,
        'frame_stack' : 4,
        'embedding_name' : 'fixed',
        'hidden_dim': 1200,
        'num_layers': 4,
        'num_filters': 32,
        'in_channels': 3,
        'num_shared_layers': 1,
        'train_visual_encoder': False,
        'num_frozen_layers': 0,
        'use_ss': False,
        'use_state_encoder': False,
        'state_hidden_dim': 100,
        'state_encoder_output_dim': 250,
        'train_policy': True,
        'train_state_encoder': False,
        'train_inv': False,
        'ss_visual_encoder_lr':3e-4,
        'ss_state_encoder_lr': 3e-4,
        'ss_inv_lr': 3e-4,
        'bc_beta': 0.99,
        "task": "pick_place_sugar_box",
        'robot_name': 'xarm6_allegro_modified_finger',
        'use_visual_obs': True,
        'adapt': False,
        "eval_freq": args.eval_freq,
        "eval_start_epoch": args.eval_start_epoch,
        "eval_only": args.eval_only,
        "ckpt": args.ckpt
    }
    args = argument_dependecy_checker(args)

    main(args)
