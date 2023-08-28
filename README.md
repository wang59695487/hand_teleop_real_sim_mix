# TODO
- Augment in simulation action spaces. hand_teleop/player/player.py L523.

# Usage
## Demo replay
export PYTHONPATH=.

Replay demos with ResNet50 as the vision backbone:
```bash
python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=resnet50 
python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=MoCo50 
python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=vit_b_16 
python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf
python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=clip_RN50

python hand_teleop/player/play_multiple_demonstrations_aug.py --backbone-type=regnet_y_3_2gf --delta-ee-pose-bound=0.0005 --out-folder=real_sim_mix/baked_data/pick_place_sugar_box_image_kinematic_aug

python hand_teleop/player/play_multiple_demonstrations_aug.py --backbone-type=regnet_y_3_2gf --delta-ee-pose-bound=0.0005 --out-folder=real_sim_mix/baked_data/pick_place_tomato_soup_can_image_kinematic_aug

python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf --delta-ee-pose-bound=0.0005 --out-folder=real_sim_mix/baked_data/dclaw_image_aug_only

python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf --delta-ee-pose-bound=0.0005 --out-folder=real_sim_mix/baked_data/pick_place_mustard_bottle_aug_img_only

python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pick_place_mustard_bottle --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=real_sim_mix/baked_data/pick_place_mustard_bottle --task-name=pick_place --object-name=mustard_bottle

```
Currently, these following backbones are supported:
- resnet50
- MoCo50
- regnet_y_1_6gf
- convnext_base
- vit_b_16 
- vit_b_32   / 3332
- vit_l_32 /4356
- swin_v2_b
- clip_ViT-B/16
- clip_RN50

## Training and evaluation
```bash
python main/train.py \
    --demo-folder=sim/baked_data/xarm_less_random_pick_place_mustard_bottle \
    --backbone-type=resnet50 \
    --eval-freq=200
```

python main/train.py \
    --demo-folder=sim/baked_data/xarm_less_random_pick_place_mustard_bottle \
    --backbone-type=MoCo50 \
    --eval-freq=100

python main/train.py \
    --demo-folder=sim/baked_data/xarm_less_random_pick_place_mustard_bottle \
    --backbone-type=regnet_y_1_6gf \
    --eval-freq=100

python main/train.py \
    --demo-folder=sim/baked_data/xarm_less_random_pick_place_mustard_bottle \
    --backbone-type=regnet_y_3_2gf \
    --eval-freq=100

python main/train.py \
    --demo-folder=sim/baked_data/xarm_less_random_pick_place_mustard_bottle \
    --backbone-type=regnet_y_8gf \
    --eval-freq=100

python main/train.py \
    --demo-folder=sim/baked_data/xarm_less_random_pick_place_mustard_bottle \
    --backbone-type=vit_b_32 \
    --eval-only \
    --ckpt=logs/Best_models/epoch_1100.pt

python main/train.py \
    --demo-folder=real/baked_data/pick_place_mustard_bottle\
    --backbone-type=regnet_y_3_2gf\
    --eval-freq=100 

nohup python main/train_real_sim.py \
    --demo-folder=real_sim_mix/baked_data/pick_place_mustard_bottle_aug_img_only \
    --backbone-type=regnet_y_3_2gf\
    --eval-freq=100 > ./logs 2>&1 &


The `eval-freq` argument specifies the frequency of evaluating and saving the model. which is 200 epochs in this case.