import torch
import numpy as np
import glob
import os
import pickle
import sys
from datetime import datetime
from functools import partial
import time

import imageio
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn, optim
from tqdm import tqdm
from transforms3d.quaternions import quat2axangle
from typing import Optional

from main.eval import apply_IK_get_real_action
from model import Agent
from hand_teleop.player.vec_player import VecPlayer
from hand_teleop.render.render_player import RenderPlayer


class VecTrainer:

    def __init__(self, args):
        self.args = args

        self.epoch_start = 0

        self.demos_train, self.demos_val = self.load_data(args)

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

        self.num_render_workers = args.n_renderers
        self.vec_player: Optional[VecPlayer] = None

    def load_data(self, args):
        demo_paths = sorted(glob.glob(os.path.join(args.demo_folder,
                                                   "*.pickle")))
        demos = []
        for path in demo_paths[:50]:
            with open(path, "rb") as f:
                cur_demo = pickle.load(f)
                # cur_demo["data"] = [{k: cur_demo["data"][k][i] for k in cur_demo["data"].keys()}
                #                     for i in range(len(cur_demo["data"]["simulation"]))]
                # # TODO: fake data, fix it later
                # for i in range(len(cur_demo["data"])):
                #     cur_demo["data"][i]["action"] = np.random.random(size=(22,)).astype(np.float32)
                demos.append(cur_demo)
        train_idx, val_idx = train_test_split(list(range(len(demos))),
                                              test_size=args.val_pct, random_state=args.seed)
        demos_train = [demos[i] for i in train_idx]
        demos_val = [demos[i] for i in val_idx]

        return demos_train, demos_val

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

    def _train_epoch(self):
        loss_train = 0
        if self.args.finetune_backbone:
            self.model.policy_net.train()
        else:
            self.model.train()

        for i in tqdm(range(0, len(self.demos_train))):
            self.vec_player.load_player_data([self.demos_train[i]] * self.num_render_workers)

            data_len = self.vec_player.data_lengths[0]
            assert self.args.batch_size % self.num_render_workers == 0

            beg = 0
            while beg < data_len:
                feature_tensor = torch.empty([self.args.batch_size, self.args.vis_dims // self.args.window_size])
                tic = time.time()
                for k in range(self.args.batch_size // self.num_render_workers):
                    end = min(beg + self.num_render_workers, data_len)
                    beg = end - self.num_render_workers
                    indices = np.arange(beg, end, dtype=int)
                    self.vec_player.set_sim_data_async(indices)
                    self.vec_player.set_sim_data_wait()
                    self.vec_player.render_async()
                    image_dict = self.vec_player.render_wait()
                    images = image_dict["Color"][:, 0, :, :, :3].permute((0, 3, 1, 2))
                    # feature_tensor[beg:beg + self.num_render_workers, :] = self.model.get_image_feats(images)
                    beg += self.num_render_workers
                tac = time.time()
                print(f"Extract feature with {feature_tensor.shape} takes {tac - tic}s.")

                # Fetch robot qpos and action
                robot_qpos = []
                actions = []
                for k in range(self.num_render_workers):
                    robot_qpos.append(self.demos_train[i + k]["data"][indices[k]]["robot_qpos"])
                for k in range(self.num_render_workers):
                    actions.append(self.demos_train[i + k]["data"][indices[k]]["action"])

                robot_qpos = torch.from_numpy(np.stack(robot_qpos, axis=0)).to(self.args.device)
                actions = torch.from_numpy(np.stack(actions[self.args.window_size - 1:], axis=0)).to(self.args.device)

                # TODO: sim only here, fix it later
                actions_pred = self.model(images, robot_qpos, "sim")
                loss = self.criterion(actions_pred, actions)
                loss.backward()
                loss_train += loss.cpu().item()

            # gradient accumulation check
            if (i + 1) % self.args.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i >= 5 and self.args.debug:
                break

        loss_train /= len(self.demos_train)

        return loss_train

    # @torch.no_grad()
    # def _eval_epoch(self):
    #     loss_val = 0
    #     self.model.eval()
    #
    #     for i in tqdm(range(len(self.demos_val))):
    #         demo_len = len(self.demos_val[i]["data"])
    #         player = self.init_player(self.demos_val[i])
    #         images = []
    #         beg = 0
    #         while beg < demo_len:
    #             end = min(beg + self.args.batch_size, demo_len)
    #             images = torch.stack([self.render_single_frame(player, t)
    #                                   for t in range(beg, end)]).permute((0, 3, 1, 2))
    #             robot_qpos = torch.from_numpy(np.stack([
    #                 self.demos_val[i]["data"][t]["robot_qpos"]
    #                 for t in range(beg, end)])).to(self.args.device)
    #             actions = torch.from_numpy(np.stack([
    #                 self.demos_val[i]["data"][t]["action"]
    #                 for t in range(beg + self.args.window_size - 1, end)])) \
    #                 .to(self.args.device)
    #
    #             # TODO: sim only here, fix it later
    #             actions_pred = self.model(images, robot_qpos, "sim")
    #             loss = self.criterion(actions_pred, actions)
    #             loss_val += loss.cpu().item()
    #
    #             beg = end
    #
    #         if i >= 5 and self.args.debug:
    #             break
    #
    #     loss_val /= len(self.demos_val)
    #     return loss_val

    def train(self):
        best_success = 0

        player_create_fn = [partial(RenderPlayer.from_demo, demo=self.demos_train[0],
                                    robot_name="xarm6_allegro_modified_finger") for i in range(self.num_render_workers)]
        self.vec_player = VecPlayer(player_create_fn)

        for i in range(self.epoch_start, self.args.epochs):
            loss_train = self._train_epoch()
            # loss_val = self._eval_epoch()
            # metrics = {
            #     "loss/train": loss_train,
            #     "loss/val": loss_val,
            #     "epoch": i,
            # }

            self.save_checkpoint("latest")

            # if (i + 1) % self.args.eval_freq == 0\
            #         and (i + 1) >= self.args.eval_beg:
            #     self.save_checkpoint(i + 1)
            #     avg_success = self.eval_in_env(self.args, i + 1,
            #         self.args.eval_x_steps, self.args.eval_y_steps)
            #     metrics.update({
            #         "avg_success": avg_success,
            #     })

            #     if avg_success > best_success:
            #         self.save_checkpoint("best")
            #         best_success = avg_success

            # if not self.args.wandb_off:
            #     wandb.log(metrics)
