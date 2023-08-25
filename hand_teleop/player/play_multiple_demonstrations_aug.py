import os
import numpy as np
import pickle
import imageio
import copy
from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as T

from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv

from hand_teleop.player.player import *
from hand_teleop.player.player_augmentation import *
from hand_teleop.player.play_multiple_demonstrations import stack_and_save_frames
from hand_teleop.player.randomization_utils import *
from hand_teleop.real_world import lab

from main.feature_extractor import generate_features, generate_feature_extraction_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
def play_multiple_sim_aug(args):
    ################Using Augmented Sim Data################
    dataset_folder = args["out_folder"]
    os.makedirs(dataset_folder, exist_ok=True)
    # shutil.rmtree("./sim/baked_data/{}".format(dataset_folder),ignore_errors=True)

    if args['with_features']:
        assert args['backbone_type'] != None
        model, preprocess = generate_feature_extraction_model(args["backbone_type"])
        model = model.to("cuda:0")
        model.eval()

    visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[], robot_qpos=[], sim_real_label=[])
    init_obj_poses = []
    demo_files = []
    for file_name in os.listdir(args['sim_demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['sim_demo_folder'], file_name))
    print('Augmenting sim demos and creating the dataset:')
    print('---------------------')
    np.random.seed(20220824)
    for demo_id, file_name in enumerate(demo_files):
        print(file_name)
        num_test = 0

        with open(file_name, 'rb') as file:
            demo = pickle.load(file)
        
        

        for i in tqdm(range(300)):

            x = np.random.uniform(-0.11,0.11)
            y = np.random.uniform(-0.11,0.11)

            if np.fabs(x) <= 0.01 and np.fabs(y) <= 0.01:
                continue
            
            all_data = copy.deepcopy(demo)
            
            info_success, visual_baked, meta_data = bake_visual_demonstration_test_augmented(all_data=all_data,init_pose_aug=sapien.Pose([x, y, 0], [1, 0, 0, 0]), retarget=args['retarget'])

            if info_success:
                print("##############SUCCESS##############")
                num_test += 1
                print("##########This is {}th try and {}th success##########".format(i+1,num_test))
                init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
                # visual_baked_demos.append(visual_baked)
                visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model=model, preprocess=preprocess)

            if num_test >= args['num_data_aug']:
                break

        sim_demo_length = len(visual_training_set['obs'])
        # since here we are using real data, we set sim_real_label = 1
        visual_training_set['sim_real_label'] = [0 for _ in range(sim_demo_length)]

        if visual_training_set['obs'] and visual_training_set['action'] and visual_training_set['robot_qpos']:
            assert len(visual_training_set['obs']) == len(visual_training_set['action'])
            print(f"Augment sim dataset for demo_{demo_id} ready:")
            print('----------------------')
            print("Number of datapoints: {}".format(len(visual_training_set['obs'])))
            print("Shape of observations: {}".format(visual_training_set['obs'][0].shape))
            print("Action dimension: {}".format(len(visual_training_set['action'][0])))
            print("Robot_qpos dimension: {}".format(len(visual_training_set['robot_qpos'][0])))
            dataset_path = "{}/{}_dataset_demo_{}_aug_{}.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""),demo_id, args['num_data_aug'])

            with open(dataset_path,'wb') as file:
                pickle.dump(visual_training_set, file)
                
        print('dataset is saved in the folder: ./real_sim_mix/baked_data/{}'.format(dataset_folder))
        meta_data_path = "{}/{}_meta_data.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
        meta_data['init_obj_poses'] = init_obj_poses
        with open(meta_data_path,'wb') as file:
            pickle.dump(meta_data, file)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--backbone-type", required=True)
    parser.add_argument("--delta-ee-pose-bound", default="0.0005", type=float)
    parser.add_argument("--sim-folder", default=None)
    parser.add_argument("--real-folder", default=None)
    parser.add_argument("--out-folder", required=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Since we are doing online retargeting currently, 'retargeting' argument should be False all the time in this file.
    # It might be better to save each frame one by one if you need the images itself. you can save it all as one file if you are just using the features.
    args = parse_args()

    args = {
        #'demo_folder' : './sim/raw_data/xarm/less_random/pick_place_mustard_bottle_0.73',
        #'sim_demo_folder': None,
        'sim_demo_folder' : './sim/raw_data/xarm/less_random/pick_place_mustard_bottle',
        #'sim_demo_folder' : './sim/raw_data/xarm/less_random/pick_place_tomato_soup_can',
        #'sim_demo_folder' : './sim/raw_data/xarm/less_random/pick_place_sugar_box',

        #'real_demo_folder': None,
        'real_demo_folder' : './real/raw_data/pick_place_mustard_bottle',
        #'real_demo_folder' : './real/raw_data/pick_place_tomato_soup_can',
        #'real_demo_folder' : './real/raw_data/pick_place_sugar_box',
    
        "robot_name": "xarm6_allegro_modified_finger",
        'with_features' : True,
        'backbone_type' : args.backbone_type,
        'stack_frames' : True,
        'retarget' : False,
        'save_each_frame' : False,
        'domain_randomization': False,
        'randomization_prob': 0.2,
        'num_data_aug': 3,
        'image_augmenter': T.AugMix(),
        'kinematic_aug': 400,
        'delta_ee_pose_bound': args.delta_ee_pose_bound,
        'out_folder': args.out_folder
    }

    if args['kinematic_aug'] > 0:
        play_multiple_sim_aug(args)