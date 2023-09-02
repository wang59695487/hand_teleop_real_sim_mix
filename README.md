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

nohup python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf --delta-ee-pose-bound=0.0005 --sim-folder=/sim/raw_data/xarm/less_random/pick_place_mustard_bottle --real-folder=real/raw_data/pick_place_mustard_bottle --out-folder=real_sim_mix/baked_data/pick_place_mustard_bottle_image_aug > ./logs/play_demo 2>&1 &

nohup python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pick_place_mustard_bottle_small_scale --out-folder=real/baked_data/pick_place_mustard_bottle_small_scale --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --delta-ee-pose-bound=0.0005 --light-mode=default > ./logs/play_real_demo 2>&1 &

python hand_teleop/player/play_multiple_demonstrations.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pick_place_mustard_bottle_large_scale --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=real_sim_mix/baked_data/pick_place_mustard_bottle --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --sim-delta-ee-pose-bound=0.01 --real-delta-ee-pose-bound=0.01 --light-mode=default --img-data-aug=5
#Real_Demos: 12815 Sim_Demos: 48735

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

# 16384 32768 65536
nohup python main/train_real_sim.py \
    --demo-folder=real_sim_mix/baked_data/pick_place_mustard_bottle_image_aug \
    --backbone-type=regnet_y_3_2gf\
    --eval-freq=100 > ./logs/train_real_sim 2>&1 &

nohup python main/train_real_sim.py \
    --demo-folder=real_sim_mix/baked_data/pick_place_mb_small_scale_wo_ran_light \
    --backbone-type=regnet_y_3_2gf \
    --lr=2e-4 \
    --sim-batch-size=16384 \
    --real-batch-size=65536 \
    --num-epochs=3000 \
    --eval-freq=100 \
    --eval-start-epoch=100 \
    --val-ratio=0.05 > ./logs/train_real_sim 2>&1 &

nohup python main/train_real_sim.py \
    --demo-folder=real_sim_mix/baked_data/pick_place_mustard_bottle \
    --backbone-type=regnet_y_3_2gf \
    --lr=2e-5 \
    --sim-batch-size=40000 \
    --real-batch-size=10000 \
    --num-epochs=2000 \
    --eval-freq=100 \
    --val-ratio=0.05 \
    --eval-start-epoch=400 > ./logs/train_real_sim 2>&1 &

python main/train_real_sim.py \
    --demo-folder=real/baked_data/pick_place_mustard_bottle_small_scale_10\
    --backbone-type=regnet_y_3_2gf \
    --lr=2e-4 \
    --real-batch-size=50000 \
    --num-epochs=3000 \
    --eval-freq=100 \
    --val-ratio=0.05 \
    --eval-start-epoch=100


The `eval-freq` argument specifies the frequency of evaluating and saving the model. which is 200 epochs in this case.

nohup python hand_teleop/player/player_augmentation.py \
      --task-name=pick_place \
      --object-name=mustard_bottle \
      --delta-object-hand-bound=0.002 \
      --delta-ee-pose-bound=0.0005 \
      --detection-bound=0.25 \
      --seed=1 \
      --frame-skip=1 > ./logs/play_aug2 2>&1 &

nohup python hand_teleop/player/player_augmentation.py \
      --task-name=pick_place \
      --object-name=sugar_box \
      --delta-object-hand-bound=0.002 \
      --delta-ee-pose-bound=0.00025 \
      --detection-bound=0.25 \
      --seed=1 \
      --frame-skip=1 > ./logs/play_aug2 2>&1 &

nohup python hand_teleop/player/player_augmentation.py \
      --task-name=pick_place \
      --object-name=tomato_soup_can \
      --delta-object-hand-bound=0.002 \
      --delta-ee-pose-bound=0.0005 \
      --detection-bound=0.25 \
      --seed=1 \
      --frame-skip=1 > ./logs/play_aug3 2>&1 &