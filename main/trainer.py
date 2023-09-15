import gc
import glob
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from transforms3d.quaternions import quat2axangle

from data import HandTeleopDataset
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from hand_teleop.player.player import DcLawEnvPlayer, PickPlaceEnvPlayer
from hand_teleop.real_world import lab
from main.eval import apply_IK_get_real_action
from model import Agent


class Trainer:

    def __init__(self, args):
        self.args = args

        self.epoch_start = 0

        self.load_data(args)

        self.model = Agent(args).to(args.device)

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

    def load_data(self, args):
        demo_paths = sorted(glob.glob(os.path.join(args.demo_folder,
            "*.pickle")))
        train_idx, val_idx = train_test_split(list(range(len(demo_paths))),
            test_size=args.val_pct, random_state=args.seed)
        demo_paths_train = [demo_paths[i] for i in train_idx]
        demo_paths_val = [demo_paths[i] for i in val_idx]


        with open(demo_paths_train[0], "rb") as f:
            sample_demo = pickle.load(f)
        self.player = self.init_player(sample_demo)
        self.meta_data = sample_demo["meta_data"]
        arm_dof = self.player.env.robot.dof
        self.init_robot_qpos = sample_demo["data"][0]["robot_qpos"]\
            [:arm_dof]

        ds_train = HandTeleopDataset(demo_paths_train)
        ds_val = HandTeleopDataset(demo_paths_val)
        self.dl_train = HandTeleopDataset.get_dataloader(ds_train,
            self.args.demo_batch_size, True, self.args.n_workers)
        self.dl_val = HandTeleopDataset.get_dataloader(ds_val,
            self.args.demo_batch_size, False, self.args.n_workers)

    def init_player(self, demo):
        meta_data = deepcopy(demo["meta_data"])
        # task_name = meta_data["env_kwargs"]["task_name"]
        # meta_data["env_kwargs"].pop("task_name")
        # meta_data["task_name"] = self.args.task
        robot_name = self.args.robot
        data = demo["data"]
        use_visual_obs = True
        if "finger_control_params" in meta_data.keys():
            finger_control_params = meta_data["finger_control_params"]
        if "root_rotation_control_params" in meta_data.keys():
            root_rotation_control_params = meta_data["root_rotation_control_params"]
        if "root_translation_control_params" in meta_data.keys():
            root_translation_control_params = meta_data["root_translation_control_params"]
        if "robot_arm_control_params" in meta_data.keys():
            robot_arm_control_params = meta_data["robot_arm_control_params"]            

        # Create env
        env_params = meta_data["env_kwargs"]
        if "task_name" in env_params:
            env_params.pop("task_name")
        env_params["robot_name"] = robot_name
        env_params["use_visual_obs"] = use_visual_obs
        env_params["use_gui"] = False

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"

        if robot_name == "mano":
            env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
        else:
            env_params["zero_joint_pos"] = None

        if "init_obj_pos" in meta_data["env_kwargs"].keys():
            env_params["init_obj_pos"] = meta_data["env_kwargs"]["init_obj_pos"]

        if "init_target_pos" in meta_data["env_kwargs"].keys():
            env_params["init_target_pos"] = meta_data["env_kwargs"]["init_target_pos"]

        if self.args.task == "pick_place":
            env = PickPlaceRLEnv(**env_params)
        elif self.args.task == "dclaw":
            env = DClawRLEnv(**env_params)
        else:
            raise NotImplementedError

        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for joint in env.robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
        env.rl_step = env.simple_sim_step

        env.reset()

        real_camera_cfg = {
            "relocate_view": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT,
            fov=lab.fov, resolution=(224, 224))
        }
        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info}}
        env.setup_visual_obs_config(camera_info)

        # Player
        if self.args.task == "pick_place":
            player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
        elif self.args.task == "dclaw":
            player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
        else:
            raise NotImplementedError

        return player

    def load_demo(self, demo):
        self.player.data = demo["data"]
        self.player.meta_data = demo["meta_data"]

    def replay_demo(self, player):
        demo_len = len(player.data)
        robot_pose = player.env.robot.get_pose()
        arm_dof = player.env.arm_dof

        player.scene.unpack(player.get_sim_data(0))
        for _ in range(player.env.frame_skip):
            player.scene.step()
        palm_pose = player.env.ee_link.get_pose()
        palm_pose = robot_pose.inv() * palm_pose
        hand_qpos = player.env.robot.get_drive_target()[arm_dof:]
        actions = []

        for i in range(1, demo_len):
            player.scene.unpack(player.get_sim_data(i))

            palm_pose_next = player.env.ee_link.get_pose()
            palm_pose_next = robot_pose.inv() * palm_pose_next
            palm_delta_pose = palm_pose.inv() * palm_pose_next
            delta_axis, delta_angle = quat2axangle(palm_delta_pose.q)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
                delta_axis = -delta_axis
            delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
            delta_pose = np.concatenate([
                palm_pose_next.p - palm_pose.p,
                delta_axis_world * delta_angle
            ])

            action = np.concatenate([delta_pose * 100, hand_qpos])
            actions.append(action)

            palm_pose = palm_pose_next
            hand_qpos = player.env.robot.get_drive_target()[arm_dof:]

        return actions

    def render_single_frame(self, idx):
        self.player.scene.unpack(self.player.get_sim_data(idx))
        image = self.player.env.get_observation()["relocate_view-rgb"].detach().clone()
        ee_pose = self.player.env.ee_link.get_pose()
        ee_pose = np.concatenate([ee_pose.p, ee_pose.q])

        return image, ee_pose

    def validate_actions(self, player, actions, video_path):
        player.scene.unpack(player.get_sim_data(0))
        video = []

        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        robot_arm_control_params = player.meta_data["finger_control_params"]
        finger_control_params = player.meta_data["robot_arm_control_params"]
        for joint in player.env.robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
        player.env.rl_step = player.env.simple_sim_step

        for i in tqdm(range(len(actions))):
            real_action = apply_IK_get_real_action(actions[i], player.env,
                player.env.robot.get_qpos(), True)
            player.env.step(real_action)
            image = player.env.get_observation()["relocate_view-rgb"]
            video.append(image.cpu().numpy())

        video = (np.stack(video) * 255).astype(np.uint8)
        imageio.mimsave(video_path, video, fps=120)

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
        batch_cnt = 0
        if self.args.finetune_backbone:
            self.model.policy_net.train()
        else:
            self.model.train()

        for _, demos in enumerate(tqdm(self.dl_train)):
            for i in tqdm(range(len(demos))):
                demo_len = len(demos[i]["data"])
                self.load_demo(demos[i])

                beg = 0
                while beg < demo_len:
                    beg = min(beg, demo_len - self.args.window_size - 1)
                    end = min(beg + self.args.step_batch_size, demo_len)
                    rendered_data = [self.render_single_frame(t)
                        for t in range(beg, end)]
                    images = torch.stack([x[0] for x in rendered_data])\
                        .permute((0, 3, 1, 2))
                    # ee_poses = torch.from_numpy(np.stack([x[1]
                    #     for x in rendered_data]))
                    robot_qpos = torch.from_numpy(np.stack([
                        demos[i]["data"][t]["robot_qpos"]
                        for t in range(beg, end)])).to(self.args.device)
                    # robot_qpos = torch.cat([robot_qpos, ee_poses], dim=-1)\
                    #     .to(self.args.device)
                    actions = torch.from_numpy(np.stack([
                        demos[i]["data"][t]["action"]
                        for t in range(beg + self.args.window_size - 1, end)]))\
                        .to(self.args.device)

                    # TODO: sim only here, fix it later
                    actions_pred = self.model(images, robot_qpos, "sim")
                    loss = self.criterion(actions_pred, actions)
                    loss.backward()
                    loss_train += loss.detach().cpu().item()
                    batch_cnt += 1

                    beg = end

                # gradient accumulation check
                if (i + 1) % self.args.grad_acc == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if _ >= 5 and self.args.debug:
                break

            if not self.args.wandb_off:
                wandb.log({
                    "running_loss": loss.detach().cpu().item(),
                    "total_steps": self.cur_epoch * len(self.dl_train) + _
                })

        loss_train /= batch_cnt

        return loss_train

    @torch.no_grad()
    def _eval_epoch(self):
        loss_val = 0
        batch_cnt = 0
        self.model.eval()

        for _, demos in enumerate(tqdm(self.dl_val)):
            for i in range(len(demos)):
                demo_len = len(demos[i]["data"])
                self.load_demo(demos[i])

                beg = 0
                while beg < demo_len:
                    beg = min(beg, demo_len - self.args.window_size - 1)
                    end = min(beg + self.args.step_batch_size, demo_len)
                    rendered_data = [self.render_single_frame(t)
                        for t in range(beg, end)]
                    images = torch.stack([x[0] for x in rendered_data])\
                        .permute((0, 3, 1, 2))
                    ee_poses = torch.from_numpy(np.stack([x[1]
                        for x in rendered_data]))
                    robot_qpos = torch.from_numpy(np.stack([
                        demos[i]["data"][t]["robot_qpos"]
                        for t in range(beg, end)])).to(self.args.device)
                    # robot_qpos = torch.cat([robot_qpos, ee_poses], dim=-1)\
                    #     .to(self.args.device)
                    actions = torch.from_numpy(np.stack([
                        demos[i]["data"][t]["action"]
                        for t in range(beg + self.args.window_size - 1, end)]))\
                        .to(self.args.device)

                    # TODO: sim only here, fix it later
                    actions_pred = self.model(images, robot_qpos, "sim")
                    loss = self.criterion(actions_pred, actions)
                    loss_val += loss.detach().cpu().item()
                    batch_cnt += 1

                    beg = end

            if _ >= 5 and self.args.debug:
                break

        loss_val /= batch_cnt

        return loss_val

    @torch.no_grad()
    def eval_in_env(self, epoch, x_steps, y_steps):
        self.model.eval()
        
        meta_data = deepcopy(self.meta_data)
        robot_name = self.args.robot
        # task_name = meta_data['task_name']
        task_name = meta_data["env_kwargs"]["task_name"]
        if 'randomness_scale' in meta_data["env_kwargs"].keys():
            randomness_scale = meta_data["env_kwargs"]['randomness_scale']
        else:
            randomness_scale = 1
        rotation_reward_weight = 0
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
        if "task_name" in env_params:
            del env_params["task_name"]
        env_params['robot_name'] = robot_name
        env_params['use_visual_obs'] = True
        env_params['use_gui'] = False

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"

        if robot_name == "mano":
            env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]

        if 'init_obj_pos' in meta_data["env_kwargs"].keys():
            env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
            object_pos = meta_data["env_kwargs"]['init_obj_pos']

        if 'init_target_pos' in meta_data["env_kwargs"].keys():
            env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
            target_pos = meta_data["env_kwargs"]['init_target_pos']

        if task_name == 'pick_place':
            env = PickPlaceRLEnv(**env_params)
        elif task_name == 'dclaw':
            env = DClawRLEnv(**env_params)
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
            "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
        }

        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)

        env.robot.set_qpos(self.init_robot_qpos)

        eval_idx = 0
        progress = tqdm(total=x_steps * y_steps)
        metrics = {"avg_success": 0}
        if self.args.task == "dclaw":
            metrics["avg_angle"] = 0

        for x in np.linspace(-0.08, 0.08, x_steps):        # -0.08 0.08 /// -0.05 0
            for y in np.linspace(0.22, 0.28, y_steps):  # 0.12 0.18 /// 0.12 0.32
                video = []
                max_angle = 0
                success = False
                object_p = np.array([x, y, env.manipulated_object.get_pose().p[-1]])
                object_pos = sapien.Pose(p=object_p, q=env.manipulated_object.get_pose().q)
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
                env.robot.set_qpos(self.init_robot_qpos)
                if self.args.task == "pick_place":
                    env.manipulated_object.set_pose(object_pos)
                for _ in range(10*env.frame_skip):
                    env.scene.step()

                obs = env.get_observation()

                for i in range(self.args.max_eval_steps):
                    video.append(obs["relocate_view-rgb"].cpu().numpy())
                    robot_qpos = env.robot.get_qpos().astype(np.float32)
                    ee_pose = env.ee_link.get_pose()
                    ee_pose = np.concatenate([ee_pose.p, ee_pose.q]).astype(np.float32)
                    robot_qpos = np.concatenate([robot_qpos, ee_pose])

                    if i == 0:
                        image_tensor = obs["relocate_view-rgb"].permute((2, 0, 1))\
                            [None].repeat(self.args.window_size, 1, 1, 1)
                        robot_qpos_tensor = torch.from_numpy(robot_qpos)\
                            [None].repeat(self.args.window_size, 1).to(self.args.device)
                    elif i == 1:
                        image_tensor[1:, ...] = obs["relocate_view-rgb"].permute((2, 0, 1))
                        robot_qpos_tensor[1:, :] = torch.from_numpy(robot_qpos)\
                            .to(self.args.device)
                    elif i == 2:
                        image_tensor[2:, ...] = obs["relocate_view-rgb"].permute((2, 0, 1))
                        robot_qpos_tensor[2:, :] = torch.from_numpy(robot_qpos)\
                            .to(self.args.device)
                    elif i == 3:
                        image_tensor[3:, ...] = obs["relocate_view-rgb"].permute((2, 0, 1))
                        robot_qpos_tensor[3:, :] = torch.from_numpy(robot_qpos)\
                            .to(self.args.device)
                    else:
                        image_tensor[1:] = image_tensor[:-1].clone()
                        robot_qpos_tensor[1:] = robot_qpos_tensor[:-1].clone()
                        image_tensor[0] = obs["relocate_view-rgb"].permute((2, 0, 1))
                        robot_qpos_tensor[0] = torch.from_numpy(robot_qpos)\
                            .to(self.args.device)

                    # TODO: sim only, fix later
                    action = self.model.get_action(image_tensor, robot_qpos_tensor, "sim")

                    real_action = apply_IK_get_real_action(action, env,
                        env.robot.get_qpos(), use_visual_obs=True)

                    next_obs, _, _, info = env.step(real_action)
                    if self.args.task == "pick_place":
                        if epoch != "best":
                            info_success = info["is_object_lifted"] and info["success"]
                        else:
                            info_success = info["success"]
                    elif self.args.task == "dclaw":
                        info_success = info["success"]
                        max_angle = max(max_angle, info["object_total_rotate_angle"])

                    success = success or info_success
                    if success:
                        break

                    obs = next_obs

                metrics["avg_success"] += int(success)
                video = (np.stack(video) * 255).astype(np.uint8)
                #If it did not lift the object, consider it as 0.25 success
                if task_name == "pick_place":
                    if epoch != "best" and info["success"]:
                        metrics["avg_success"] += 0.25

                    is_lifted = info["is_object_lifted"]
                    video_path = os.path.join(self.log_dir,
                        f"epoch_{epoch}_{eval_idx}_{success}_{is_lifted}.mp4")

                elif task_name == "dclaw":
                    video_path = os.path.join(self.log_dir,
                        f"epoch_{epoch}_{eval_idx}_{success}_{max_angle}.mp4")
                    metrics["avg_angle"] += max_angle
                imageio.mimsave(video_path, video, fps=120)
                eval_idx += 1
                progress.update()

        metrics["avg_success"] /= eval_idx
        if task_name == "dclaw":
            metrics["avg_angle"] /= eval_idx
        progress.close()

        return metrics

    def train(self):
        best_success = 0
        best_angle = 0

        for i in range(self.epoch_start, self.args.epochs):
            self.cur_epoch = i
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
                env_metrics = self.eval_in_env(i + 1,
                    self.args.eval_x_steps, self.args.eval_y_steps)
                metrics.update(env_metrics)

                if self.args.task == "pick_place":
                    if metrics["avg_success"] > best_success:
                        self.save_checkpoint("best")
                        best_success = metrics["avg_success"]
                else:
                    if metrics["avg_angle"] > best_angle:
                        self.save_checkpoint("best")
                        best_angle = metrics["avg_angle"]

            if not self.args.wandb_off:
                wandb.log(metrics)
