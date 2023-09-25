import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from vector_trainer import VecTrainer


def parse_args():
    parser = ArgumentParser()

    # required
    parser.add_argument("--demo-folder", required=True)
    parser.add_argument("--seed", required=True, type=int)

    # data
    parser.add_argument("--robot", default="xarm6_allegro_modified_finger")
    parser.add_argument("--robot-dof", default=22, type=int)
    parser.add_argument("--arm-dof", default=6, type=int)

    # model
    parser.add_argument("--backbone", default="regnet_y_3_2gf")
    parser.add_argument("--n-enc-layers", default=4, type=int)
    parser.add_argument("--n-dec-layers", default=7, type=int)
    parser.add_argument("--n-heads", default=8, type=int)
    parser.add_argument("--n-queries", default=30, type=int)
    parser.add_argument("--hidden-dims", default=256, type=int)
    parser.add_argument("--forward-dims", default=3200, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--action-dims", default=22, type=int)
    parser.add_argument("--latent-dims", default=1024, type=int)
    parser.add_argument("--vision-dims", default=1512, type=int)
    parser.add_argument("--qpos-dims", default=29, type=int)
    parser.add_argument("--get-obj-pose", action="store_true")

    # training
    parser.add_argument("--max-lr", default=1e-3, type=float)
    parser.add_argument("--min-lr", default=2e-6, type=float)
    parser.add_argument("--wd-coef", default=1e-2, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--grad-acc", default=1, type=int)
    parser.add_argument("--small-scale", action="store_true")
    parser.add_argument("--w-kl-loss", default=1, type=float)
    parser.add_argument("--n-renderers", default=4, type=int)
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--min-demo-len", default=400, type=int)
    parser.add_argument("--rnd-lvl", default=1, type=int)

    # evaluation
    parser.add_argument("--eval-x-steps", default=4, type=int)
    parser.add_argument("--eval-y-steps", default=5, type=int)
    parser.add_argument("--final-x-steps", default=10, type=int)
    parser.add_argument("--final-y-steps", default=10, type=int)
    parser.add_argument("--val-pct", default=0.1, type=float)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-beg", default=0, type=int)
    parser.add_argument("--eval-freq", default=5, type=int)
    parser.add_argument("--max-eval-steps", default=1200, type=int)
    parser.add_argument("--w-action-ema", default=0.01, type=float)

    # others
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n-workers", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--wandb-off", action="store_true")
    parser.add_argument("--one-demo", action="store_true")

    args = parser.parse_args()

    if os.path.basename(args.demo_folder).startswith("pick_place"):
        args.task = "pick_place"
    elif os.path.basename(args.demo_folder).startswith("dclaw"):
        args.task = "dclaw"

    if args.debug:
        args.n_workers = 0
        args.eval_freq = 1
        args.eval_x_steps = 2
        args.eval_y_steps = 2
        args.eval_beg = 0
        args.max_eval_steps = 10

    if args.get_obj_pose:
        args.qpos_dims += 7

    return args


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    set_rng_seed(args.seed)
    trainer = VecTrainer(args)

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    if not args.eval_only:
        trainer.train()
        trainer.load_checkpoint(f"{trainer.log_dir}/model_best.pth")

    avg_success = trainer.eval_in_env("best",
        trainer.args.final_x_steps, trainer.args.final_y_steps)
    print(f"Average success rate: {avg_success:.4f}")
    wandb.log({"final_success": avg_success})

    wandb.finish()


if __name__ == "__main__":
    main()
