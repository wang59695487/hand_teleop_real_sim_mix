import gc
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
sys.path.append("/kaiming-fast-vol-1/workspace/hand_teleop_real_sim_mixture")

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from data import HandTeleopDataset, collate_fn
from hand_teleop.player.player import PickPlaceEnvPlayer
from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from losses import DomainClfLoss
from hand_teleop.real_world import lab
from main.eval import apply_IK_get_real_action
from model import Agent


class Trainer:

    def __init__(self, args):
        self.args = args
        self.epoch_start = 0

        self.dl_train, self.dl_val = self.init_dataloaders(args)

        self.model = Agent(args).to(args.device)

        if args.grad_rev:
            self.criterion = DomainClfLoss(0.5)
        else:
            self.criterion = nn.MSELoss()
        if args.finetune_backbone:
            self.optimizer = optim.AdamW(self.model.parameters(),
                args.lr, weight_decay=args.wd_coef)
        else:
            self.optimizer = optim.AdamW(self.model.policy_net.parameters(),
                args.lr, weight_decay=args.wd_coef)

        self.start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.args.ckpt is None:
            self.log_dir = f"logs/{self.args.task}_{self.args.backbone}_{self.start_time}"
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = os.path.dirname(self.args.ckpt)

        if not args.wandb_off:
            wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
            wandb.login(key=wandb_cfg.key)
            wandb.init(
                project="hand-teleop",
                name=os.path.basename(self.log_dir),
                config=self.args
            )

    def init_dataloaders(self, args):
        with open(os.path.join(args.demo_folder,
                f"{args.backbone}_dataset.pickle"), "rb") as file:
            data = pickle.load(file)
        random_order = np.random.permutation(len(data["action"]))

        obs = np.stack(data["obs"])[random_order]
        next_obs = np.stack(data["next_obs"])[random_order]
        robot_qpos = np.stack(data["robot_qpos"])[random_order]
        actions = np.stack(data["action"])[random_order]
        domains = np.stack(data["sim_real_label"])[random_order]
        val_cutoff = int(len(obs) * args.val_pct)
        data_train = {
            "obs": obs[val_cutoff:],
            "next_obs": next_obs[val_cutoff:],
            "robot_qpos": robot_qpos[val_cutoff:],
            "actions": actions[val_cutoff:],
            "domains": domains,
        }
        data_val = {
            "obs": obs[:val_cutoff],
            "next_obs": next_obs[:val_cutoff],
            "robot_qpos": robot_qpos[:val_cutoff],
            "actions": actions[:val_cutoff],
            "domains": domains,
        }
        ds_train = HandTeleopDataset(data_train)
        ds_val = HandTeleopDataset(data_val)
        dl_train = DataLoader(ds_train, args.batch_size, True,
            num_workers=args.n_workers, collate_fn=collate_fn)
        dl_val = DataLoader(ds_val, args.batch_size, False,
            num_workers=args.n_workers, collate_fn=collate_fn)

        return dl_train, dl_val

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint["model"])
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not ckpt_path.endswith("best.pth"):
            self.epoch_start = int(os.path.basename(ckpt_path)\
                .split(".")[0].split("_")[1]) - 1
        self.log_dir = os.path.dirname(ckpt_path)

    def save_checkpoint(self, epoch):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.log_dir, f"model_{epoch}.pth")
        torch.save(state_dict, save_path)

    def _train_epoch(self):
        loss_train = 0
        if self.args.grad_rev:
            action_loss_train = 0
            domain_loss_train = 0
        self.model.train()

        for i, sample in enumerate(tqdm(self.dl_train)):
            obs = sample["obs"].to(self.args.device)
            robot_qpos = sample["robot_qpos"].to(self.args.device)
            actions = sample["action"].to(self.args.device)

            if self.args.grad_rev:
                actions_pred, domains_pred = self.model(robot_qpos, obs)
                domains = sample["domain"].to(self.args.device)
                loss, action_loss, domain_loss = self.criterion(actions_pred,
                    actions, domains_pred, domains)
            else:
                actions_pred = self.model(robot_qpos, obs)
                loss = self.criterion(actions_pred, actions)

            loss.backward()
            loss_train += loss.cpu().item()
            if self.args.grad_rev:
                action_loss_train += action_loss.cpu().item()
                domain_loss_train += domain_loss.cpu().item()

            # gradient accumulation check
            if (i + 1) % self.args.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i >= 5 and self.args.debug:
                break

        loss_train /= len(self.dl_train)
        if self.args.grad_rev:
            action_loss_train /= len(self.dl_train)
            domain_loss_train /= len(self.dl_train)
            return loss_train, action_loss_train, domain_loss_train
        else:
            return loss_train

    @torch.no_grad()
    def _eval_epoch(self):
        loss_val = 0
        if self.args.grad_rev:
            action_loss_val = 0
            domain_loss_val = 0
        self.model.eval()

        for i, sample in enumerate(tqdm(self.dl_val)):
            obs = sample["obs"].to(self.args.device)
            robot_qpos = sample["robot_qpos"].to(self.args.device)
            actions = sample["action"].to(self.args.device)

            if self.args.grad_rev:
                actions_pred, domains_pred = self.model(robot_qpos, obs)
                domains = sample["domain"].to(self.args.device)
                loss, action_loss, domain_loss = self.criterion(actions_pred,
                    actions, domains_pred, domains)
            else:
                actions_pred = self.model(robot_qpos, obs)
                loss = self.criterion(actions_pred, actions)

            loss_val += loss.cpu().item()
            if self.args.grad_rev:
                action_loss_val += action_loss.cpu().item()
                domain_loss_val += domain_loss.cpu().item()

            if i >= 5 and self.args.debug:
                break

        loss_val /= len(self.dl_val)
        if self.args.grad_rev:
            action_loss_val /= len(self.dl_train)
            domain_loss_val /= len(self.dl_train)
            return loss_val, action_loss_val, domain_loss_val
        else:
            return loss_val

    @torch.no_grad()
    def eval_in_env(self, args, epoch, x_steps, y_steps):
        self.model.eval()

        with open(os.path.join(args.demo_folder,
                f"{args.backbone}_meta_data.pickle"), "rb") as file:
            meta_data = pickle.load(file)
        # --Create Env and Robot-- #
        robot_name = args.robot
        # task_name = meta_data['task_name']
        task_name = "pick_place"
        if 'randomness_scale' in meta_data["env_kwargs"].keys():
            randomness_scale = meta_data["env_kwargs"]['randomness_scale']
        else:
            randomness_scale = 1
        rotation_reward_weight = 0
        use_visual_obs = True
        if 'allegro' in robot_name:
            if 'finger_control_params' in meta_data.keys():
                finger_control_params = meta_data['finger_control_params']
            if 'root_rotation_control_params' in meta_data.keys():
                root_rotation_control_params = meta_data['root_rotation_control_params']
            if 'root_translation_control_params' in meta_data.keys():
                root_translation_control_params = meta_data['root_translation_control_params']
            if 'robot_arm_control_params' in meta_data.keys():
                robot_arm_control_params = meta_data['robot_arm_control_params']            

        env_params = meta_data["env_kwargs"]
        env_params['robot_name'] = robot_name
        env_params['use_visual_obs'] = True
        env_params['use_gui'] = False

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"

        if robot_name == "mano":
            env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]

        if 'init_obj_pos' in meta_data["env_kwargs"].keys():
            print('Found initial object pose')
            env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
            object_pos = meta_data["env_kwargs"]['init_obj_pos']

        if 'init_target_pos' in meta_data["env_kwargs"].keys():
            print('Found initial target pose')
            env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
            target_pos = meta_data["env_kwargs"]['init_target_pos']

        if task_name == 'pick_place':
            env = PickPlaceRLEnv(**env_params)
        elif task_name == 'hammer':
            env = HammerRLEnv(**env_params)
        elif task_name == 'table_door':
            env = TableDoorRLEnv(**env_params)
        elif task_name == 'insert_object':
            env = InsertObjectRLEnv(**env_params)
        elif task_name == 'mug_flip':
            env = MugFlipRLEnv(**env_params)
        else:
            raise NotImplementedError
        env.seed(0)
        env.reset()

        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
                elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                    joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
                else:
                    joint.set_drive_property(*(finger_control_params), mode="acceleration")
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
                else:
                    joint.set_drive_property(*(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step

        real_camera_cfg = {
            "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(69.4), resolution=(224, 224))
        }
        
        if task_name == 'table_door':
            camera_cfg = {
            "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                    right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
            }           
        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)

        with open('{}/{}_dataset.pickle'.format(args.demo_folder, args.backbone), 'rb') as file:
            dataset = pickle.load(file)
            if 'state' in dataset.keys():
                init_robot_qpos = dataset['state'][0][-7-env.robot.dof:-7]
                state_shape = len(dataset['state'][0])
                concatenated_obs_shape = None
                # print('State shape: {}'.format(state_shape))
            else:
                init_robot_qpos = dataset['robot_qpos'][0][:env.robot.dof]
                concatenated_obs_shape = len(dataset['obs'][0])
                state_shape = None
            action_shape = len(dataset['action'][0])

        env.robot.set_qpos(init_robot_qpos)

        eval_idx = 0
        avg_success = 0
        progress = tqdm(total=x_steps * y_steps)
        for x in np.linspace(-0.1, 0.1, x_steps): # -0.15, 0.18, 0.03  # -0.1, 0.0, 0.02
            for y in np.linspace(0.2, 0.3, y_steps): # 0.05, 0.2, 0.05 # 0.1, 0.2, 0.02
                video = []
                idx = np.random.randint(len(meta_data['init_obj_poses']))
                sampled_pos = meta_data['init_obj_poses'][idx]
                object_p = np.array([x, y, sampled_pos.p[-1]])
                object_pos = sapien.Pose(p=object_p, q=sampled_pos.q)
                print('Object Pos: {}'.format(object_pos))
                env.reset()
                if "free" in robot_name:
                    for joint in env.robot.get_active_joints():
                        name = joint.get_name()
                        if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                            joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
                        elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                            joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
                        else:
                            joint.set_drive_property(*(finger_control_params), mode="acceleration")
                    env.rl_step = env.simple_sim_step
                elif "xarm" in robot_name:
                    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
                    for joint in env.robot.get_active_joints():
                        name = joint.get_name()
                        if name in arm_joint_names:
                            joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
                        else:
                            joint.set_drive_property(*(1 * finger_control_params), mode="force")
                    env.rl_step = env.simple_sim_step                
                env.robot.set_qpos(init_robot_qpos)
                env.manipulated_object.set_pose(object_pos)
                for _ in range(10*env.frame_skip):
                    env.scene.step()
                obs = env.get_observation()
                success = False
                for i in range(args.max_eval_steps):
                    video.append(obs["relocate_view-rgb"])
                    robot_qpos = np.concatenate([
                        env.robot.get_qpos(),
                        env.ee_link.get_pose().p,
                        env.ee_link.get_pose().q
                    ])
                    if i == 0:
                        cur_images = np.stack([
                            obs["relocate_view-rgb"],
                            obs["relocate_view-rgb"],
                            obs["relocate_view-rgb"],
                            obs["relocate_view-rgb"],
                        ])
                        cur_states = np.stack([
                            robot_qpos,
                            robot_qpos,
                            robot_qpos,
                            robot_qpos,
                        ])
                    elif i == 1:
                        cur_images = np.stack([
                            cur_images[0],
                            obs["relocate_view-rgb"],
                            obs["relocate_view-rgb"],
                            obs["relocate_view-rgb"],
                        ])
                        cur_states = np.stack([
                            cur_states[0],
                            robot_qpos,
                            robot_qpos,
                            robot_qpos,
                        ])
                    elif i == 2:
                        cur_images = np.stack([
                            cur_images[0],
                            cur_images[1],
                            obs["relocate_view-rgb"],
                            obs["relocate_view-rgb"],
                        ])
                        cur_states = np.stack([
                            cur_states[0],
                            cur_states[1],
                            robot_qpos,
                            robot_qpos,
                        ])
                    else:
                        cur_images = np.stack([
                            cur_images[-3],
                            cur_images[-2],
                            cur_images[-1],
                            obs["relocate_view-rgb"],
                        ])
                        cur_states = np.stack([
                            cur_states[-3],
                            cur_states[-2],
                            cur_states[-1],
                            robot_qpos,
                        ])
                    cur_images_tensor = torch.from_numpy(cur_images)[None, ...].permute((0, 1, 4, 2, 3)).to(args.device)
                    cur_states_tensor = torch.from_numpy(cur_states).to(args.device)
                    action = self.model.get_action(cur_states_tensor, cur_images_tensor)
                    # NOTE For new version, uncomment below!
                    real_action = apply_IK_get_real_action(action, env, env.robot.get_qpos(), use_visual_obs=True)

                    # next_obs, reward, done, _ = env.step(action)
                    # NOTE For new version, uncomment below!
                    next_obs, reward, done, info = env.step(real_action)
                    if epoch != "best":
                        info_success = info["is_object_lifted"] and info["success"]
                    else:
                        info_success = info["success"]

                    success = success or info_success
                    if success:
                        break

                    obs = deepcopy(next_obs)

                avg_success += int(success)
                video = (np.stack(video) * 255).astype(np.uint8)

                is_lifted = info["is_object_lifted"]
                video_path = os.path.join(self.log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{is_lifted}.mp4")
                imageio.mimsave(video_path, video, fps=120)
                eval_idx += 1
                progress.update()

        avg_success /= eval_idx
        progress.close()

        del cur_images_tensor
        del cur_states_tensor
        torch.cuda.empty_cache()
        del cur_images
        del cur_states
        gc.collect()

        print(f"avg_success in epoch {epoch}: {avg_success:.4f}")
        return avg_success

    def train(self):
        best_success = 0

        for i in range(self.epoch_start, self.args.epochs):
            if self.args.grad_rev:
                loss_train, action_loss_train, domain_loss_train =\
                    self._train_epoch()
                loss_val, action_loss_val, domain_loss_val = self._eval_epoch()
                metrics = {
                    "loss/train": loss_train,
                    "loss/val": loss_val,
                    "action_loss/train": action_loss_train,
                    "action_loss/val": action_loss_val,
                    "domain_loss/train": domain_loss_train,
                    "domain_loss/val": domain_loss_val,
                    "epoch": i,
                }
            else:
                loss_train = self._train_epoch()
                loss_val = self._eval_epoch()
                metrics = {
                    "loss/train": loss_train,
                    "loss/val": loss_val,
                    "epoch": i,
                }

            self.save_checkpoint("latest")

            if (i + 1) % self.args.eval_freq == 0\
                    and (i + 1) >= self.args.eval_beg:
                self.save_checkpoint(i + 1)
                avg_success = self.eval_in_env(self.args, i + 1,
                    self.args.eval_x_steps, self.args.eval_y_steps)
                metrics.update({
                    "avg_success": avg_success,
                })

                if avg_success > best_success:
                    self.save_checkpoint("best")
                    best_success = avg_success

            if not self.args.wandb_off:
                wandb.log(metrics)
