import glob
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Optional

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn, optim
from tqdm import tqdm
from transforms3d.quaternions import quat2axangle

from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from hand_teleop.player.player import DcLawEnvPlayer, PickPlaceEnvPlayer
from hand_teleop.player.vec_player import VecPlayer
from hand_teleop.real_world import lab
from hand_teleop.render.render_player import RenderPlayer
from main.eval import apply_IK_get_real_action
from model import Agent


class VecTrainer:

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
            self.log_dir = f"logs/{self.args.task}_{self.start_time}"
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

        self.num_render_workers = args.n_renderers
        self.vec_player: Optional[VecPlayer] = None

    def load_data(self, args):
        demo_paths = sorted(glob.glob(os.path.join(args.demo_folder,
            "*.pickle")))
        with open(demo_paths[0], "rb") as f:
            self.sample_demo = pickle.load(f)
        self.meta_data = self.sample_demo["meta_data"]
        self.init_robot_qpos = self.sample_demo["data"][0]["robot_qpos"]\
            [:self.args.robot_dof]
        train_idx, val_idx = train_test_split(list(range(len(demo_paths))),
            test_size=args.val_pct, random_state=args.seed)
        self.demo_paths_train = [demo_paths[i] for i in train_idx]
        self.demo_paths_val = [demo_paths[i] for i in val_idx]

    def load_demo(self, demo_path):
        with open(demo_path, "rb") as f:
            demo = pickle.load(f)

        return demo

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint["model"])
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not ckpt_path.endswith("best.pth"):
            self.epoch_start = int(os.path.basename(ckpt_path) \
                                   .split(".")[0].split("_")[1]) - 1
        self.log_dir = os.path.dirname(ckpt_path)

    def save_checkpoint(self, epoch):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.log_dir, f"model_{epoch}.pth")
        torch.save(state_dict, save_path)

    @torch.no_grad()
    def eval_in_env(self, epoch, x_steps, y_steps):
        self.model.eval()
        meta_data = self.meta_data
        # task_name = meta_data["env_kwargs"]["task_name"]
        # meta_data["env_kwargs"].pop("task_name")
        # meta_data["task_name"] = self.args.task
        robot_name = self.args.robot
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

        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)

        env.robot.set_qpos(self.init_robot_qpos)

        eval_idx = 0
        progress = tqdm(total=x_steps * y_steps)
        metrics = {"avg_success": 0}
        if self.args.task == "dclaw":
            metrics["avg_angle"] = 0

        if self.args.task == "pick_place":
            x_beg, x_end = -0.08, 0.08
            y_beg, y_end = 0.22, 0.28
        else:
            x_beg, x_end = -0.05, 0.05
            y_beg, y_end = 0.05, 0.05

        for x in np.linspace(x_beg, x_end, x_steps):        # -0.08 0.08 /// -0.05 0
            for y in np.linspace(y_beg, y_end, y_steps):  # 0.12 0.18 /// 0.12 0.32
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
                if self.args.task == "pick_place":
                    if epoch != "best" and info["success"]:
                        metrics["avg_success"] += 0.25

                    is_lifted = info["is_object_lifted"]
                    video_path = os.path.join(self.log_dir,
                        f"epoch_{epoch}_{eval_idx}_{success}_{is_lifted}.mp4")

                elif self.args.task == "dclaw":
                    video_path = os.path.join(self.log_dir,
                        f"epoch_{epoch}_{eval_idx}_{success}_{max_angle}.mp4")
                    metrics["avg_angle"] += max_angle
                imageio.mimsave(video_path, video, fps=120)
                eval_idx += 1
                progress.update()

        metrics["avg_success"] /= eval_idx
        if self.args.task == "dclaw":
            metrics["avg_angle"] /= eval_idx
        progress.close()

        return metrics

    def _train_epoch(self):
        loss_train = 0
        batch_cnt = 0
        if self.args.finetune_backbone:
            self.model.train()
        else:
            self.model.vision_net.eval()
            self.model.policy_net.train()

        for i in tqdm(range(0, len(self.demo_paths_train))):
            cur_demo = self.load_demo(self.demo_paths_train[i])
            self.vec_player.load_player_data([cur_demo] * self.num_render_workers)

            data_len = int(self.vec_player.data_lengths[0])
            assert self.args.batch_size % self.num_render_workers == 0

            beg = 0
            while beg < data_len:
                image_tensor = []
                robot_qpos_tensor = []
                action_tensor = []

                t_beg = time.time()
                for k in range(self.args.batch_size // self.num_render_workers):
                    end = min(beg + self.num_render_workers, data_len)
                    beg = end - self.num_render_workers
                    indices = np.arange(beg, end, dtype=int)
                    self.vec_player.set_sim_data_async(indices)
                    self.vec_player.set_sim_data_wait()
                    self.vec_player.render_async()
                    image_dict = self.vec_player.render_wait()
                    images = image_dict["Color"][:, 0, :, :, :3]\
                        .permute((0, 3, 1, 2))
                    image_tensor.append(images)
                    robot_qpos_tensor.append(torch.from_numpy(np.stack([
                        cur_demo["data"][t]["robot_qpos"]
                        for t in range(beg, end)])))
                    action_tensor.append(torch.from_numpy(np.stack([
                        cur_demo["data"][t]["action"]
                        for t in range(beg, end)])))
                    beg += self.num_render_workers
                    if end == data_len:
                        break

                t_end = time.time()
                print(f"Render {self.args.batch_size} images takes {t_end - t_beg:.4f}s")

                if end == data_len and data_len % self.num_render_workers > 0:
                    res_len = self.num_render_workers - data_len % self.num_render_workers
                    image_tensor[-1] = image_tensor[-1][res_len:]
                    robot_qpos_tensor[-1] = robot_qpos_tensor[-1][res_len:]
                    action_tensor[-1] = action_tensor[-1][res_len:]
                image_tensor = torch.cat(image_tensor)
                if image_tensor.size(0) <= 4:
                    continue
                robot_qpos_tensor = torch.cat(robot_qpos_tensor).to(self.args.device)
                action_tensor = torch.cat(action_tensor)[self.args.window_size - 1:].to(self.args.device)

                # TODO: sim only here, fix it later
                t_beg = time.time()
                actions_pred = self.model(image_tensor, robot_qpos_tensor, "sim")
                loss = self.criterion(actions_pred, action_tensor)
                loss.backward()
                loss_train += loss.cpu().item()
                batch_cnt += 1
                t_end = time.time()
                print(f"Forward/backward {self.args.batch_size} images takes {t_end - t_beg:.4f}s")

                if not self.args.wandb_off:
                    wandb.log({
                        "running_loss": loss.cpu().item()
                    })

            # gradient accumulation check
            
            if (i + 1) % self.args.grad_acc == 0:
                # print(f"Demo {i}, update parameters.")
                t_beg = time.time()
                self.optimizer.step()
                self.optimizer.zero_grad()
                t_end = time.time()
                print(f"Update parameters takes {t_end - t_beg:.4f} s")

            if i >= 5 and self.args.debug:
                break

        loss_train /= batch_cnt

        return loss_train

    @torch.no_grad()
    def _eval_epoch(self):
        loss_val = 0
        batch_cnt = 0
        self.model.eval()

        for i in tqdm(range(0, len(self.demo_paths_val))):
            cur_demo = self.load_demo(self.demo_paths_val[i])
            self.vec_player.load_player_data([cur_demo] * self.num_render_workers)

            data_len = int(self.vec_player.data_lengths[0])
            assert self.args.batch_size % self.num_render_workers == 0

            beg = 0
            while beg < data_len:
                image_tensor = []
                robot_qpos_tensor = []
                action_tensor = []
                for k in range(self.args.batch_size // self.num_render_workers):
                    end = min(beg + self.num_render_workers, data_len)
                    beg = end - self.num_render_workers
                    indices = np.arange(beg, end, dtype=int)
                    self.vec_player.set_sim_data_async(indices)
                    self.vec_player.set_sim_data_wait()
                    self.vec_player.render_async()
                    image_dict = self.vec_player.render_wait()
                    images = image_dict["Color"][:, 0, :, :, :3]\
                        .permute((0, 3, 1, 2))
                    image_tensor.append(images)
                    robot_qpos_tensor.append(torch.from_numpy(np.stack([
                        cur_demo["data"][t]["robot_qpos"]
                        for t in range(beg, end)])))
                    action_tensor.append(torch.from_numpy(np.stack([
                        cur_demo["data"][t]["action"]
                        for t in range(beg, end)])))
                    beg += self.num_render_workers
                    if end == data_len:
                        break

                if end == data_len and data_len % self.num_render_workers > 0:
                    res_len = self.num_render_workers - data_len % self.num_render_workers
                    image_tensor[-1] = image_tensor[-1][res_len:]
                    robot_qpos_tensor[-1] = robot_qpos_tensor[-1][res_len:]
                    action_tensor[-1] = action_tensor[-1][res_len:]
                image_tensor = torch.cat(image_tensor)
                if image_tensor.size(0) <= 4:
                    continue
                robot_qpos_tensor = torch.cat(robot_qpos_tensor).to(self.args.device)
                action_tensor = torch.cat(action_tensor)[self.args.window_size - 1:].to(self.args.device)

                # TODO: sim only here, fix it later
                actions_pred = self.model(image_tensor, robot_qpos_tensor, "sim")
                loss = self.criterion(actions_pred, action_tensor)
                loss_val += loss.cpu().item()
                batch_cnt += 1

            if i >= 5 and self.args.debug:
                break

        loss_val /= batch_cnt

        return loss_val

    def train(self):
        best_success = 0

        player_create_fn = [partial(RenderPlayer.from_demo, demo=self.sample_demo,
            robot_name="xarm6_allegro_modified_finger") for i in range(self.num_render_workers)]
        self.vec_player = VecPlayer(player_create_fn)

        for i in range(self.epoch_start, self.args.epochs):
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

                if metrics["avg_success"] > best_success:
                    self.save_checkpoint("best")
                    best_success = metrics["avg_success"]

            if not self.args.wandb_off:
                wandb.log(metrics)
