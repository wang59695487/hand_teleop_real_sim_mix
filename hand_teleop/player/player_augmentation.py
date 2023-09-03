import shutil
from typing import Dict, Any, Optional, List

import numpy as np
import sapien.core as sapien
import transforms3d
import pickle
import os
import imageio
from tqdm import tqdm
import copy
from argparse import ArgumentParser

import sys
sys.path.append("/home/yigit/project/hand_teleop_real_sim_mix")

from hand_teleop.utils.common_robot_utils import LPFilter
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.real_world import lab
from hand_teleop.player.player import *

def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping ** 2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ \
                 np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos

def generate_sim_aug(args, all_data, init_pose_aug, retarget=False, frame_skip=1):

    from pathlib import Path
    # Recorder
    #path = f"./sim/raw_data/{task_name}_{object_name}/{object_name}_{demo_index:04d}.pickle"
    #path = "sim/raw_data/pick_place_tomato_soup_can/tomato_soup_can_0011.pickle"
    #path = "sim/raw_data/pick_place_sugar_box/sugar_box_0050.pickle"
    #path = "sim/raw_data/dclaw/dclaw_3x_0001.pickle"
    #all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    data = all_data["data"]
    use_visual_obs = True
    if not retarget:
        robot_name = meta_data["robot_name"]
    else:
        robot_name = "allegro_hand_free"
    if 'allegro' in robot_name:
        if 'finger_control_params' in meta_data.keys():
            finger_control_params = meta_data['finger_control_params']
        if 'root_rotation_control_params' in meta_data.keys():
            root_rotation_control_params = meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in meta_data.keys():
            root_translation_control_params = meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in meta_data.keys():
            robot_arm_control_params = meta_data['robot_arm_control_params']       

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = False
    # env_params = dict(object_name=meta_data["env_kwargs"]['object_name'], object_scale=meta_data["env_kwargs"]['object_scale'], robot_name=robot_name, 
    #                  rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=meta_data["env_kwargs"]['randomness_scale'], 
    #                  use_visual_obs=use_visual_obs, use_gui=False)
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None
    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'dclaw':
        env = DClawRLEnv(**env_params)
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

    if not retarget:
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
            
    env.reset()
    
    real_camera_cfg = {
        "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
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

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])

        meta_data["env_kwargs"]['init_obj_pos'] = init_pose_aug * meta_data["env_kwargs"]['init_obj_pos']
        meta_data["env_kwargs"]['init_target_pos'] = init_pose_aug * meta_data["env_kwargs"]['init_target_pos']

    elif task_name == 'dclaw':
        player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration(init_pose_aug = init_pose_aug)

    env.reset()
    player.scene.unpack(player.get_sim_data(0))
    
    # for _ in range(player.env.frame_skip):
    #     player.scene.step()
    # if player.human_robot_hand is not None:
    #     player.scene.remove_articulation(player.human_robot_hand.robot)

    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])

    robot_pose = env.robot.get_pose()

    ### Aug obj pose ###
    env.manipulated_object.set_pose(meta_data["env_kwargs"]['init_obj_pos'])

    if task_name =='pick_place':
        env.plate.set_pose(meta_data["env_kwargs"]['init_target_pos'])
    
    data = []
    rgb_pics = []
    
    is_hand_grasp = False
    ee_pose = baked_data["ee_pose"][0]
    hand_qpos_prev = baked_data["action"][0][env.arm_dof:]
    dist_object_hand_prev = np.linalg.norm(env.manipulated_object.pose.p - env.ee_link.get_pose().p)
    stop_frame = 0

    for idx in tqdm(range(0,len(baked_data["obs"]),frame_skip)):
        action = baked_data["action"][idx]
        ee_pose = baked_data["ee_pose"][idx]
        # NOTE: robot.get_qpos() version
        if idx < len(baked_data['obs'])-frame_skip:
            ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
            ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx!=0 else hand_qpos

            object_pose = env.manipulated_object.pose.p
            palm_pose = env.ee_link.get_pose()

            dist_object_hand = np.linalg.norm(object_pose - ee_pose_next[:3])
            delta_object_hand = dist_object_hand_prev - dist_object_hand

            if ee_pose_delta < args['delta_ee_pose_bound'] and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2:
                #print("!!!!!!!!!!!!!!!!!!!!!!skip!!!!!!!!!!!!!!!!!!!!!")
                continue
                
            else:

                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos

                if task_name =='pick_place':

                    if np.mean(handqpos2angle(delta_hand_qpos)) > 1 and dist_object_hand_prev < args['detection_bound']:
                        is_hand_grasp = True
                    
                    if dist_object_hand_prev < args['detection_bound'] and not(is_hand_grasp) and delta_object_hand < args['delta_object_hand_bound']:
                        continue

                    if env._object_target_distance() < 0.25 and object_pose[2] < 0.2:
                        hand_qpos = hand_qpos*0.9
              
                palm_pose = robot_pose.inv() * palm_pose

                palm_next_pose = sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                palm_next_pose = robot_pose.inv() * palm_next_pose

                palm_delta_pose = palm_pose.inv() * palm_next_pose
                delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
                if delta_angle > np.pi:
                    delta_angle = 2 * np.pi - delta_angle
                    delta_axis = -delta_axis
                delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
                arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
                arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                item_data = {"simulation":env.scene.pack(), 
                             "action": np.concatenate([delta_pose*100, hand_qpos]), 
                             "robot_qpos": np.concatenate([env.robot.get_qpos(),env.ee_link.get_pose().p,env.ee_link.get_pose().q])}
                data.append(item_data)
                rgb_pics.append(env.get_observation()["relocate_view-rgb"].cpu().detach().numpy())
                _, _, _, info = env.step(target_qpos)
                
                if task_name == 'pick_place':
                    info_success = info["is_object_lifted"] and env._object_target_distance() <= 0.2 and env._is_object_plate_contact()
                    dist_object_hand_prev = np.linalg.norm(env.manipulated_object.pose.p - env.ee_link.get_pose().p)
                                    
                if info_success:
                    stop_frame += 1
                
                if stop_frame == 16:
                    break
                #env.render()
    
    meta_data["env_kwargs"]['task_name'] = task_name
    augment_data = {'data': data, 'meta_data': meta_data}

    
    
    for i in range(len(rgb_pics)):
        rgb = rgb_pics[i]
        rgb_pics[i] = (rgb * 255).astype(np.uint8)

    return info_success, augment_data, rgb_pics

def generate_sim_aug_in_play_demo(args, demo, init_pose_aug):

    robot_name=args['robot_name']
    retarget=args['retarget']
    frame_skip=args['frame_skip']
    light_mode=args['light_mode']

    from pathlib import Path
    if robot_name == 'mano':
        assert retarget == False
    # Get env params
    meta_data = demo["meta_data"]
    if not retarget:    
        assert robot_name == meta_data["robot_name"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    meta_data["task_name"] = task_name
    if 'randomness_scale' in meta_data["env_kwargs"].keys():
        randomness_scale = meta_data["env_kwargs"]['randomness_scale']
    else:
        randomness_scale = 1

    data = demo["data"]
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

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = False
    # env_params = dict(object_name=meta_data["env_kwargs"]['object_name'], object_scale=meta_data["env_kwargs"]['object_scale'], robot_name=robot_name, 
    #                  rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=meta_data["env_kwargs"]['randomness_scale'], 
    #                  use_visual_obs=use_visual_obs, use_gui=False)
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None
    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'dclaw':
        env = DClawRLEnv(**env_params)
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

    if not retarget:
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
            
    env.reset()
    
    real_camera_cfg = {
        "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
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

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])

        meta_data["env_kwargs"]['init_obj_pos'] = init_pose_aug * meta_data["env_kwargs"]['init_obj_pos']
        meta_data["env_kwargs"]['init_target_pos'] = init_pose_aug * meta_data["env_kwargs"]['init_target_pos']

    elif task_name == 'dclaw':
        player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration(init_pose_aug = init_pose_aug)

    env.reset()
    player.scene.unpack(player.get_sim_data(0))
    
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])

    robot_pose = env.robot.get_pose()

    ### Aug obj pose ###
    env.manipulated_object.set_pose(meta_data["env_kwargs"]['init_obj_pos'])

    if task_name =='pick_place':
        env.plate.set_pose(meta_data["env_kwargs"]['init_target_pos'])
    
    
    is_hand_grasp = False
    ee_pose = baked_data["ee_pose"][0]
    hand_qpos_prev = baked_data["action"][0][env.arm_dof:]
    dist_object_hand_prev = np.linalg.norm(env.manipulated_object.pose.p - env.ee_link.get_pose().p)
    stop_frame = 0
    grasp_frame = 0
    visual_baked = dict(obs=[], action=[],robot_qpos=[])
    for idx in tqdm(range(0,len(baked_data["obs"]),frame_skip)):
        action = baked_data["action"][idx]
        ee_pose = baked_data["ee_pose"][idx]
        # NOTE: robot.get_qpos() version
        if idx < len(baked_data['obs'])-frame_skip:
            ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
            ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx!=0 else hand_qpos

            object_pose = env.manipulated_object.pose.p
            palm_pose = env.ee_link.get_pose()

            dist_object_hand = np.linalg.norm(object_pose - ee_pose_next[:3])
            delta_object_hand = dist_object_hand_prev - dist_object_hand

            if ee_pose_delta < args['sim_delta_ee_pose_bound'] and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2:
                #print("!!!!!!!!!!!!!!!!!!!!!!skip!!!!!!!!!!!!!!!!!!!!!")
                continue
                
            else:

                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos

                if task_name == "pick_place":
                    if np.mean(handqpos2angle(delta_hand_qpos)) > 1 and dist_object_hand_prev < args['detection_bound']:
                        is_hand_grasp = True
                
                    if dist_object_hand_prev < args['detection_bound'] and not(is_hand_grasp) and delta_object_hand < args['delta_object_hand_bound']:
                        continue

                    if env._object_target_distance() < 0.25 and object_pose[2] < 0.2:
                        hand_qpos = hand_qpos*0.9
              
                palm_pose = robot_pose.inv() * palm_pose

                palm_next_pose = sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                palm_next_pose = robot_pose.inv() * palm_next_pose

                palm_delta_pose = palm_pose.inv() * palm_next_pose
                delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
                if delta_angle > np.pi:
                    delta_angle = 2 * np.pi - delta_angle
                    delta_axis = -delta_axis
                delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
                arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
                arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                observation = env.get_observation()
                visual_baked["obs"].append(observation)
                visual_baked["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
                # Using robot qpos version
                visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                                                    env.ee_link.get_pose().p,env.ee_link.get_pose().q]))
                _, _, _, info = env.step(target_qpos)

                if task_name == "pick_place":

                    # if np.mean(handqpos2angle(delta_hand_qpos)) > 1 and dist_object_hand_prev < 0.15:
                    #     ###########################Grasping augmentation############################
                    #     for _ in range(5):
                    #         visual_baked["obs"].append(observation)
                    #         visual_baked["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
                    #         # Using robot qpos version
                    #         visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                    #                                         env.ee_link.get_pose().p,env.ee_link.get_pose().q]))
                    #         grasp_frame+=1

                    info_success = info["is_object_lifted"] and env._object_target_distance() <= 0.2 and env._is_object_plate_contact()
                    dist_object_hand_prev = np.linalg.norm(env.manipulated_object.pose.p - env.ee_link.get_pose().p)
                                    
                if info_success:
                    stop_frame += 1
                
                if stop_frame == 16:
                    break

    # if grasp_frame/len(visual_baked['obs']) < 0.1:
    #     info_success = False
    # print("grasp_frame: ", grasp_frame)
    # print("total_frame: ", len(visual_baked['obs']))

    return visual_baked, meta_data, info_success


def player_augmenting(args):

    np.random.seed(args['seed'])

    demo_files = []
    sim_data_path = f"{args['sim_demo_folder']}/{args['task_name']}_{args['object_name']}/"
    for file_name in os.listdir(sim_data_path ):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(sim_data_path, file_name))
    print('Augmenting sim demos and creating the dataset:')
    print('---------------------')

    for demo_id, file_name in enumerate(demo_files):

        num_test = 0
        # if demo_id <= 5:
        #     continue
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)

        for i in range(400):
            x = np.random.uniform(-0.11,0.11)
            y = np.random.uniform(-0.11,0.11)
            
            if np.fabs(x) <= 0.01 and np.fabs(y) <= 0.01:
                continue

            all_data = copy.deepcopy(demo)

            out_folder = f"./sim/raw_augmentation_action/{args['task_name']}_{args['object_name']}_aug/"
            os.makedirs(out_folder, exist_ok=True)

            # if len(os.listdir(out_folder)) == 0:
            #     num_test = "0001"
            # else:
            #     pkl_files = os.listdir(out_folder)
            #     last_num = sorted([int(x.replace(".pickle", "").split("_")[-1]) for x in pkl_files])[-1]
            #     num_test = str(last_num + 1).zfill(4)
                
            info_success, data, video = generate_sim_aug(args, all_data=all_data, init_pose_aug=sapien.Pose([x, y, 0], [1, 0, 0, 0]),frame_skip=args['frame_skip'], retarget=args['retarget'])
            #imageio.mimsave(f"./temp/demos/aug_{args['object_name']}/demo_{demo_id+1}_{num_test}_x{x:.2f}_y{y:.2f}.mp4", video, fps=120)
            if info_success:

                print("##############SUCCESS##############")
                num_test += 1
                print("##########This is {}th try and {}th success##########".format(i+1,num_test))

                imageio.mimsave(f"./temp/demos/aug_{args['object_name']}/demo_{demo_id+1}_{num_test}_x{x:.2f}_y{y:.2f}.mp4", video, fps=30)
                
                dataset_folder = f"{out_folder}/demo_{demo_id+1}_{num_test}.pickle"

                with open(dataset_folder,'wb') as file:
                    pickle.dump(data, file)
            
            if num_test == args['kinematic_aug']:
                break


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=20230826, type=int)
    parser.add_argument("--sim-demo-folder",default='./sim/raw_data/', type=str)
    parser.add_argument("--task-name", required=True, type=str)
    parser.add_argument("--object-name", required=True, type=str)
    parser.add_argument("--kinematic-aug", default=100, type=int)
    parser.add_argument("--frame-skip", default=1, type=int)
    parser.add_argument("--retarget", default=False, type=bool)
    parser.add_argument("--delta-object-hand-bound", default=0, type=float)
    parser.add_argument("--delta-ee-pose-bound", default=0, type=float)
    parser.add_argument("--detection-bound", default=0.25, type=float)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
     
    args = {
        'seed': args.seed,
        'sim_demo_folder' : args.sim_demo_folder,
        'task_name': args.task_name,
        'object_name': args.object_name,
        'kinematic_aug': args.kinematic_aug,
        'frame_skip': args.frame_skip,
        'delta_object_hand_bound': args.delta_object_hand_bound,
        'delta_ee_pose_bound': args.delta_ee_pose_bound,
        'detection_bound': args.detection_bound,
        'retarget': args.retarget
    }

    player_augmenting(args)
        

    
    
