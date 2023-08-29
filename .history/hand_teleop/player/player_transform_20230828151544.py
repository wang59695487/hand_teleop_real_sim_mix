import shutil
from typing import Dict, Any, Optional, List

import numpy as np
import sapien.core as sapien
import transforms3d
import pickle
from pathlib import Path
import os
import imageio

from hand_teleop.env.rl_env.base import BaseRLEnv, compute_inverse_kinematics
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv

from hand_teleop.player.player import *
from hand_teleop.utils.common_robot_utils import LPFilter
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.real_world import lab


def bake_visual_real_demonstration_test(retarget=False, index = 0):
    from pathlib import Path

    # Recorder
    shutil.rmtree('./temp/demos/player', ignore_errors=True)
    os.makedirs('./temp/demos/player')
    path = "./sim/raw_data/xarm/less_random/pick_place_mustard_bottle/mustard_bottle_0002.pickle"
    #path = "./sim/raw_data/xarm/less_random/pick_place_sugar_box/sugar_box_0001.pickle"
    #path = "sim/raw_data/xarm/less_random/pick_place_tomato_soup_can/tomato_soup_can_0001.pickle"
    #path = "sim/raw_data/xarm/less_random/dclaw/dclaw_3x_0001.pickle"

    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    #meta_data['env_kwargs']['init_target_pos'] = sapien.Pose([-0.05, -0.105, 0], [1, 0, 0, 0])
    data = all_data["data"]
    use_visual_obs = True

    #print(meta_data)
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
    env_params['use_gui'] = True
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
        print(env_params['init_target_pos'])
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
    viewer = env.render(mode="human")
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, np.pi/4)  

    # real_camera_cfg = {
    #     "relocate_view": dict( pose= ROBOT2BASE * CAM2ROBOT, fov=np.deg2rad(47.4), resolution=(320, 240))
    # }
    
    real_camera_cfg = {
        "relocate_view": dict( pose= lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
    }

    if task_name == 'table_door':
         camera_cfg = {
        "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }
            
    env.setup_camera_from_config(real_camera_cfg)
    #env.setup_camera_from_config(camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
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

    # Retargeting
    using_real = True
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    elif using_real:
        record_root  = "./real/raw_data/pick_place_mustard_bottle"
        #record_root  = "./real/raw_data/dclaw"
        
        #record_root  = "./real/raw_data/pick_place_tomato_soup_can"
        #record_root  = "./real/raw_data/pick_place_sugar_box"
        record_path = Path(record_root) / f"{index:04}.pkl"
        #path = "./real/raw_data/pick_place_mustard_bottle/0001.pkl"
        baked_data = np.load(record_path, allow_pickle=True)
        
    visual_baked = dict(obs=[], action=[])
    env.reset()

    player.scene.unpack(player.get_sim_data(0))
    # env.randomize_object_rotation()
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)
    
    #robot_base_pose = np.array([0, -0.7, 0, 0.707, 0, 0, 0.707])
    env.robot.set_qpos(baked_data[0]["teleop_cmd"])
    #print("init_qpos: ",baked_data[0]["teleop_cmd"])

    robot_pose = env.robot.get_pose()
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(-np.matmul(rotation_matrix.T,robot_pose.p),rotation_matrix.T,np.ones(3))
    
    for idx in range(len(baked_data)):

        ee_pose = env.ee_link.get_pose()
        baked_data[idx]["ee_pose"] = np.concatenate([ee_pose.p,ee_pose.q])
        #print(baked_data[idx]["ee_pose"])
        # NOTE: robot.get_qpos() version
        if idx != len(baked_data)-1:
            
            hand_qpos = baked_data[idx+1]["teleop_cmd"][env.arm_dof:]
            arm_qpos = baked_data[idx+1]["teleop_cmd"][:env.arm_dof]

            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            # env.step(target_qpos)
            env.robot.set_qpos(target_qpos)
            env.render()
    
    with record_path.open("wb") as f:
        pickle.dump(baked_data, f)

    # for i in range(len(visual_baked["obs"])):
    #     rgb = visual_baked["obs"][i]["relocate_view-rgb"]
    #     rgb_pic = (rgb * 255).astype(np.uint8)
    #     imageio.imsave("./temp/demos/player/relocate-rgb_{}.png".format(i), rgb_pic)
    
    

if __name__ == '__main__':
    # bake_demonstration_adroit()
    # bake_demonstration_allegro_test()
    # bake_demonstration_svh_test()
    # bake_demonstration_ar10_test()
    # bake_demonstration_mano()
    # bake_visual_demonstration_test()
    for i in range(15):
        bake_visual_real_demonstration_test(index = i)
