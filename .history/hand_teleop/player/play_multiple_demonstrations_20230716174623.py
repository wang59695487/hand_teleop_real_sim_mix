import os
import numpy as np
import pickle
import imageio

from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.player.player import *
from hand_teleop.player.randomization_utils import *

from main.feature_extractor import generate_features

def play_multiple(args):
    demo_files = []
    dataset_folder = '{}_{}_{}_states'.format(args['demo_folder'].split('/')[-3],args['demo_folder'].split('/')[-2],args['demo_folder'].split('/')[-1])
    shutil.rmtree("./sim/baked_data/{}".format(dataset_folder), ignore_errors=True)
    
    for file_name in os.listdir(args['demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['demo_folder'], file_name))
    print('Replaying the demos and creating the dataset:')
    print('---------------------')
    training_set = dict(obs=[], next_obs=[], action=[])
    init_obj_poses = []
    for file_name in demo_files:
        print(file_name)
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)
            baked_data, meta_data = play_one_demo(demo=demo, robot_name=args['robot_name'], domain_randomization=args['domain_randomization'], randomization_prob=args['randomization_prob'], retarget=args['retarget'])
            init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
        oracle_observations = []
        target_actions = []
        for act, obs in zip(baked_data["action"], baked_data["obs"]):
            oracle_observations.append(obs)
            target_actions.append(act)
        stacked_obs = []
        for i in range(len(oracle_observations)):
            if i==0:
                stacked_obs.append(np.concatenate((oracle_observations[i],oracle_observations[i],oracle_observations[i],oracle_observations[i])))         
            elif i==1:
                stacked_obs.append(np.concatenate((oracle_observations[i-1],oracle_observations[i],oracle_observations[i],oracle_observations[i])))         
            elif i==2:
                stacked_obs.append(np.concatenate((oracle_observations[i-2],oracle_observations[i-1],oracle_observations[i],oracle_observations[i])))         
            else:
                stacked_obs.append(np.concatenate((oracle_observations[i-3],oracle_observations[i-2],oracle_observations[i-1],oracle_observations[i])))         
        assert len(oracle_observations) == len(stacked_obs)
        stacked_next_obs = stacked_obs[1:]
        stacked_obs = stacked_obs[:-1]
        target_actions = target_actions[:-1]
        training_set['obs'].extend(stacked_obs)
        training_set['next_obs'].extend(stacked_next_obs)
        training_set['action'].extend(target_actions)

    print("Dataset ready:")
    print('----------------------')
    print("Number of datapoints: {}".format(len(training_set['obs'])))
    print("Shape of observations: {}".format(training_set['obs'][0].shape))
    print("Action dimension: {}".format(len(training_set['action'][0])))
    os.makedirs("./sim/baked_data/{}".format(dataset_folder), exist_ok=True)
    dataset_path = "./sim/baked_data/{}/dataset.pickle".format(dataset_folder)
    with open(dataset_path,'wb') as file:
        pickle.dump(training_set, file)
    print('dataset is saved in the folder: ./sim/baked_data/{}'.format(dataset_folder))
    meta_data_path = "./sim/baked_data/{}/meta_data.pickle".format(dataset_folder)
    meta_data['init_obj_poses'] = init_obj_poses
    with open(meta_data_path,'wb') as file:
        pickle.dump(meta_data, file)


def play_one_demo(demo, robot_name, domain_randomization, randomization_prob, retarget=False):
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
    rotation_reward_weight = 0
    use_visual_obs = False
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

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None

    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        print('Found initial object pose')
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']

    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        print('Found initial target pose')
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']

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
    
    if domain_randomization:
        p = np.random.rand(1)[0]
        if p <= randomization_prob:
            env = randomize_env_colors(task_name=task_name, env=env)

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

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
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
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration()
    extracted_data = dict(obs=[], action=[])
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
    for idx, (obs, qpos, state, action, ee_pose) in enumerate(zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                        baked_data["action"], baked_data["ee_pose"])):
        # NOTE: robot.get_qpos() version
        if idx != len(baked_data['obs'])-1:
            palm_pose = env.ee_link.get_pose()
            palm_pose = robot_pose.inv() * palm_pose

            ee_pose_next = baked_data["ee_pose"][idx + 1]
            palm_next_pose = sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
            palm_next_pose = robot_pose.inv() * palm_next_pose

            palm_delta_pose = palm_pose.inv() * palm_next_pose
            delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
                delta_axis = -delta_axis
            delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
            #delta_pose??
            delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

            palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
            arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
            arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]
            hand_qpos = action[env.arm_dof:]
            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            extracted_data["obs"].append(env.get_observation())
            extracted_data["action"].append(10*np.concatenate([delta_pose*100, hand_qpos]))
            env.step(target_qpos)

        # # NOTE: Old Version
        # extracted_data["obs"].append(env.get_observation())
        # extracted_data["action"].append(action)
        # env.step(action)

    return extracted_data, meta_data


def play_multiple_visual(args):
    demo_files = []
    dataset_folder = '{}_{}_{}'.format(args['demo_folder'].split('/')[-3],args['demo_folder'].split('/')[-2],args['demo_folder'].split('/')[-1])
    shutil.rmtree("./sim/baked_data/{}".format(dataset_folder),ignore_errors=True)
    if args['with_features']:
        assert args['backbone_type'] != None
    
    for file_name in os.listdir(args['demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['demo_folder'], file_name))
    print('Replaying the demos and creating the dataset:')
    print('---------------------')
    visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[])
    init_obj_poses = []
    for demo_id, file_name in enumerate(demo_files):
        print(file_name)
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)
            visual_baked, meta_data = play_one_visual_demo(demo=demo, robot_name=args['robot_name'], domain_randomization=args['domain_randomization'], randomization_prob=args['randomization_prob'], retarget=args['retarget'])
            init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
            # visual_baked_demos.append(visual_baked)
        visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args)

    if visual_training_set['obs'] and visual_training_set['action']:
        assert len(visual_training_set['obs']) == len(visual_training_set['action'])
        print("Dataset ready:")
        print('----------------------')
        print("Number of datapoints: {}".format(len(visual_training_set['obs'])))
        print("Shape of observations: {}".format(visual_training_set['obs'][0].shape))
        print("Action dimension: {}".format(len(visual_training_set['action'][0])))
        os.makedirs("./sim/baked_data/{}".format(dataset_folder), exist_ok=True)
        dataset_path = "./sim/baked_data/{}/dataset.pickle".format(dataset_folder)
        with open(dataset_path,'wb') as file:
            pickle.dump(visual_training_set, file)
    print('dataset is saved in the folder: ./sim/baked_data/{}'.format(dataset_folder))
    meta_data_path = "./sim/baked_data/{}/meta_data.pickle".format(dataset_folder)
    meta_data['init_obj_poses'] = init_obj_poses
    with open(meta_data_path,'wb') as file:
        pickle.dump(meta_data, file)


def play_one_visual_demo(demo, robot_name, domain_randomization, randomization_prob, retarget=False):
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

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = False

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None

    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        print('Found initial object pose')
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']

    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        print('Found initial target pose')
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']

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
    
    if domain_randomization:
        p = np.random.rand(1)[0]
        if p <= randomization_prob:
            env = randomize_env_colors(task_name=task_name, env=env)

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

    # Create camera
    # camera_cfg = {
    #     "relocate_view": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
    #                             right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
    # }
    camera_cfg = {
        "relocate_view": dict(position=np.array([0.25, 0.25, 0.45]), look_at_dir=np.array([-0.25, -0.25, -0.35]),
                                right_dir=np.array([-1, 1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
    }
    if task_name == 'table_door':
         camera_cfg = {
        "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }   
    env.setup_camera_from_config(camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
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
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration()
    visual_baked = dict(obs=[], action=[])
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
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(-np.matmul(rotation_matrix.T,robot_pose.p),rotation_matrix.T,np.ones(3))
    for idx, (obs, qpos, state, action, ee_pose) in enumerate(zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                        baked_data["action"], baked_data["ee_pose"])):
        # NOTE: robot.get_qpos() version
        if idx != len(baked_data['obs'])-1:
            palm_pose = env.ee_link.get_pose()
            palm_pose = robot_pose.inv() * palm_pose

            ee_pose_next = baked_data["ee_pose"][idx + 1]
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
            hand_qpos = action[env.arm_dof:]
            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            visual_baked["obs"].append(env.get_observation())
            visual_baked["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
            env.step(target_qpos)

        # # NOTE: Old Version
        # visual_baked["obs"].append(env.get_observation())
        # visual_baked["action"].append(action)
        # env.step(action)

    # # For visual obs debugging    
    # for i in range(len(visual_baked["obs"])):
    #     rgb = visual_baked["obs"][i]["relocate_view-rgb"]
    #     rgb_pic = (rgb * 255).astype(np.uint8)
    #     imageio.imsave("./temp/demos/single/relocate-rgb_{}.png".format(i), rgb_pic)

    return visual_baked, meta_data

def stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args):
    if args['with_features']:
        if 'state' in visual_training_set.keys():
            visual_training_set.pop('state')
            visual_training_set.pop('next_state')
        visual_demo_with_features = generate_features(visual_baked=visual_baked, backbone_type=args['backbone_type'], stack_frames=args['stack_frames'])
        stacked_next_obs = visual_demo_with_features['obs'][1:]
        stacked_obs = visual_demo_with_features['obs'][:-1]
        actions = visual_demo_with_features['action'][:-1]
        visual_training_set['obs'].extend(stacked_obs)
        visual_training_set['next_obs'].extend(stacked_next_obs)
        visual_training_set['action'].extend(actions)
    else:
        rgb_imgs = []
        robot_states = []
        target_actions = []
        for act, obs in zip(visual_baked["action"], visual_baked["obs"]):
            rgb_imgs.append(obs["relocate_view-rgb"])
            robot_states.append(obs["state"])
            target_actions.append(act)
        stacked_frames = []
        stacked_states = []
        for i in range(len(rgb_imgs)):
            if i==0:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0))
                stacked_states.append(np.concatenate((robot_states[i],robot_states[i],robot_states[i],robot_states[i])))         
            elif i==1:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0))
                stacked_states.append(np.concatenate((robot_states[i-1],robot_states[i],robot_states[i],robot_states[i])))         
            elif i==2:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0))
                stacked_states.append(np.concatenate((robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i])))         
            else:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]), axis=-1), -1, 0))        
                stacked_states.append(np.concatenate((robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i])))         
        assert len(rgb_imgs) == len(stacked_frames)
        assert len(robot_states) == len(stacked_states)
        stacked_next_frames = stacked_frames[1:]
        stacked_frames = stacked_frames[:-1]
        stacked_next_states = stacked_states[1:]
        stacked_states = stacked_states[:-1]
        target_actions = target_actions[:-1]
        visual_training_set['obs'].extend(stacked_frames)
        visual_training_set['next_obs'].extend(stacked_next_frames)
        visual_training_set['state'].extend(stacked_states)
        visual_training_set['next_state'].extend(stacked_next_states)
        visual_training_set['action'].extend(target_actions)

    if args['save_each_frame']:
        dataset_folder = './sim/baked_data/{}'.format(dataset_folder)
        os.makedirs(dataset_folder,exist_ok=True)
        for idx in range(len(visual_training_set['obs'])):
            frame = {}
            frame['obs'] = visual_training_set['obs'][idx]
            frame['next_obs'] = visual_training_set['next_obs'][idx]
            frame['action'] = visual_training_set['action'][idx]
            if 'state' in visual_training_set.keys():
                frame['state'] = visual_training_set['state'][idx]
                frame['next_state'] = visual_training_set['next_state'][idx]
            with open(os.path.join(dataset_folder,'{}_{}.pickle'.format(demo_id, idx)),'wb') as file:
                pickle.dump(frame, file)
        visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[])
    
    return visual_training_set        

if __name__ == '__main__':
    # Since we are doing online retargeting currently, 'retargeting' argument should be False all the time in this file.
    # It might be better to save each frame one by one if you need the images itself. you can save it all as one file if you are just using the features.
    args = {
        'demo_folder' : './sim/raw_data/xarm/less_random/pick_place_mustard_bottle',
        # 'demo_folder' : './sim/raw_data/xarm/less_random/pick_place_tomato_soup_can',
        # # 'robot_name' : 'allegro_hand_xarm6_wrist_mounted_face_down',
        "robot_name": "xarm6_allegro_modified_finger",
        'use_true_states' : False,
        'with_features' : True,
        'backbone_type' : 'MoCo50',
        'stack_frames' : True,
        'retarget' : False,
        'save_each_frame' : False,
        'domain_randomization': False,
        'randomization_prob': 0.2,
    }
    if args['use_true_states']:
        play_multiple(args)
    else:
        play_multiple_visual(args)
