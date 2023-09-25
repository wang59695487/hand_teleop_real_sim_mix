import numpy as np
import torch
import os
import pickle
from copy import deepcopy
from tqdm import tqdm

from feature_extractor import generate_feature_extraction_model
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv

from hand_teleop.player.randomization_utils import *
from hand_teleop.player.player import *
from hand_teleop.real_world import task_setting
from hand_teleop.real_world import lab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(32)

def apply_IK_get_real_action(action,env,qpos,use_visual_obs):
    if not use_visual_obs:
        action = action/10
    # action = action/10    
    delta_pose = np.squeeze(action)[:env.arm_dof]/100
    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(qpos[:env.arm_dof])
    arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
    arm_qpos = arm_qvel + qpos[:env.arm_dof]
    hand_qpos = np.squeeze(action)[env.arm_dof:]
    target_qpos = np.concatenate([arm_qpos, hand_qpos])
    return target_qpos

def eval_in_env(args, agent, log_dir, epoch, x_steps, y_steps):
    with open("{}/{}_meta_data.pickle".format(args["sim_dataset_folder"], args["backbone_type"].replace("/", "")),'rb') as file:
        meta_data = pickle.load(file)
    
    # --Create Env and Robot-- #
    robot_name = args["robot_name"]
    task_name = meta_data['task_name']
    rotation_reward_weight = 0
    use_visual_obs = args['use_visual_obs']
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
    env_params['light_mode'] = "default" if args['randomness_rank'] < 3 else "random"

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

    if args['use_visual_obs']:

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

    with open('{}/{}_dataset.pickle'.format(args["sim_dataset_folder"], args["backbone_type"].replace("/", "")), 'rb') as file:
        print('sim_dataset_folder: {}'.format(args["sim_dataset_folder"]))
        dataset = pickle.load(file)
        print(dataset.keys())
        if 'state' in dataset.keys():
            init_robot_qpos = dataset['state'][0][-7-env.robot.dof:-7]
            state_shape = len(dataset['state'][0])
            concatenated_obs_shape = None
            # print('State shape: {}'.format(state_shape))
        else:
            if task_name == "pick_place":
                init_robot_qpos = [0, (-45/180)*np.pi, 0, 0, (45/180)*np.pi, (-90/180)*np.pi] + [0] * 16
            elif task_name == "dclaw":
                init_robot_qpos = [0, (20/180)*np.pi, -(85/180)*np.pi, 0, (112/180)*np.pi, -np.pi / 2] + [0] * 16
            concatenated_obs_shape = len(dataset['obs'][0])
            state_shape = None
        action_shape = len(dataset['action'][0])

    if concatenated_obs_shape != None:
        feature_extractor, preprocess = generate_feature_extraction_model(backbone_type=args['backbone_type'])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

    env.robot.set_qpos(init_robot_qpos)

    eval_idx = 0
    avg_success = 0
    progress = tqdm(total=x_steps * y_steps)

    # since in simulation, we always use simulated data, so sim_real_label is always 0
    sim_real_label = [0]
    var_object = [0,0] if args['randomness_rank'] not in [3,6] else [0.05,0.1]    
    for x in np.linspace(-0.1-var_object[0], 0.1+var_object[0], x_steps):        # -0.08 0.08 /// -0.05 0
        for y in np.linspace(0.2-var_object[1], 0.3+var_object[1], y_steps):  # 0.12 0.18 /// 0.12 0.32
            video = []
            idx = np.random.randint(len(meta_data['init_obj_poses']))
            sampled_pos = meta_data['init_obj_poses'][idx]
            object_p = np.array([x, y, sampled_pos.p[-1]])
            object_pos = sapien.Pose(p=object_p, q=sampled_pos.q)
            print('Object Pos: {}'.format(object_pos))

            ########### Add Plate Randomness ############
            if args['randomness_rank'] in [2,3,6] and task_name == "pick_place":
                ########### Randomize the plate pose ############
                var_plate = 0.05 if args['randomness_rank'] in [2] else 0.1
                print("############################Randomize the plate pose##################")
                x2 = np.random.uniform(-var_plate, var_plate)
                y2 = np.random.uniform(-var_plate, var_plate)
                plate_random_plate = sapien.Pose([-0.005+x2, -0.12+y2, 0],[1,0,0,0]) 
                env.plate.set_pose(plate_random_plate)
                print('Target Pos: {}'.format(plate_random_plate))
            
            ############## Add Texture Randomness ############
            if args['randomness_rank'] in [4,6] :
                #env.random_light(args['randomness_rank']-2)
                env.generate_random_object_texture(2)
            
            ############## Add Light Randomness ############
            if args['randomness_rank'] in [5,6] :
                env.random_light(2)
                
            
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
            max_time_step = 1200
            action_dim = 22
            all_time_actions = torch.zeros([max_time_step, max_time_step+args['num_queries'], action_dim]).cuda()

            for i in range(max_time_step):
                #video.append(obs["relocate_view-rgb"])
                video.append(obs["relocate_view-rgb"].cpu().detach().numpy())
                if concatenated_obs_shape != None:
                    assert args['adapt'] == False
                    if args['use_visual_obs']:
                        img = torch.moveaxis(obs["relocate_view-rgb"],-1,0)[None, ...]
                        img = preprocess(img)
                        img = img.to(device)
                        #print(img.size())
                        with torch.no_grad():
                            feature = feature_extractor(img)
                            feature = torch.reshape(feature, (-1,))
                        robot_qpos = np.concatenate([env.robot.get_qpos(),env.ee_link.get_pose().p,env.ee_link.get_pose().q])
                feature = feature[None,...]
                robot_qpos = robot_qpos[None,...]
                robot_qpos = torch.from_numpy(robot_qpos).to(device)
                action = agent.evaluate(feature, robot_qpos)
                all_time_actions[[i], i:i+args['num_queries']] = action
                actions_for_curr_step = all_time_actions[:, i]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                raw_action = raw_action.cpu().detach().numpy()
                real_action = apply_IK_get_real_action(raw_action, env, env.robot.get_qpos(), use_visual_obs=use_visual_obs)

                next_obs, reward, done, info = env.step(real_action)

                if task_name == "pick_place":
                    info_success = info["is_object_lifted"] and info["success"]
                elif task_name == "dclaw":
                    info_success = info["success"]
                
                success = success or info_success
                if success:
                    break
                # env.render()

                obs = deepcopy(next_obs)
            
            avg_success += int(success)
            video = (np.stack(video) * 255).astype(np.uint8)

            #only save video if success or in the final_success evaluation
            #if success or epoch == "best":
            if task_name == "pick_place":
                is_lifted = info["is_object_lifted"]
                video_path = os.path.join(log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{is_lifted}.mp4")
            elif task_name == "dclaw":
                total_angle = info["object_total_rotate_angle"]
                video_path = os.path.join(log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{total_angle}.mp4")
            #imageio version 2.28.1 imageio-ffmpeg version 0.4.8 scikit-image version 0.20.0
            imageio.mimsave(video_path, video, fps=120)
            eval_idx += 1
            progress.update()

    avg_success /= eval_idx
    progress.close()
    
    print("avg_success in epoch", epoch, ":", avg_success)
    return avg_success