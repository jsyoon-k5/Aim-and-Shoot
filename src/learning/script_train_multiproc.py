import sys, os, copy

sys.path.append("..")

from agent.agents import *
from configs.simulation import *
from learning.sac_base import MultiProcessingModulatedSACTrain

import argparse
from typing import Union

parser = argparse.ArgumentParser(description='Training option')

parser.add_argument('--exp', type=int, default=0)
parser.add_argument('--batch_sz', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.03125e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--train_freq', type=int, default=1)
parser.add_argument('--ent', type=str, default='auto')
parser.add_argument('--target_ent', type=str, default='auto')
parser.add_argument('--un', type=int, default=512)
parser.add_argument('--dn', type=int, default=3)
parser.add_argument('--concat', type=str, default='all', choices=["all", "input", "hidden"])
parser.add_argument('--cpu', type=int, default=16)

parser.add_argument('--save_freq', type=int, default=2e6)
parser.add_argument('--eval_freq', type=int, default=5e4)
parser.add_argument('--eval_ep', type=int, default=2048)

parser.add_argument('--load', type=str, default='')
parser.add_argument('--ckpt', type=str, default='65000000')
parser.add_argument('--name', type=str, default='modmp')


args = parser.parse_args()

try:
    entropy_value = float(args.ent)
except:
    entropy_value = args.ent
try:
    entropy_target = float(args.target_ent)
except:
    entropy_target = args.target_ent

if args.concat == 'all':
    cc_layer = [i for i in range(args.dn+1)]
elif args.concat == "hidden":
    cc_layer = [i for i in range(0, args.dn+1)]
else:   # Input
    cc_layer = [0]

if __name__ == "__main__":
    if args.exp == 0:
        config = copy.deepcopy(USER_CONFIG_3)
        trainer = MultiProcessingModulatedSACTrain(
            model_name='full',
            env_class=VariableEnvTimeNoise,
            env_setting=config,
            num_cpu=args.cpu
        )
    elif args.exp == 1:
        config = copy.deepcopy(USER_CONFIG_3_BASE)
        trainer = MultiProcessingModulatedSACTrain(
            model_name='base',
            env_class=VariableEnvTimeNoiseBase,     # No Gaze
            env_setting=config,
            num_cpu=args.cpu
        )
    elif args.exp == 2:
        config = copy.deepcopy(USER_CONFIG_3)
        trainer = MultiProcessingModulatedSACTrain(
            model_name='base2',
            env_class=VariableEnvTimeNoiseBase2,     # No efference copy
            env_setting=config,
            num_cpu=args.cpu
        )
    elif args.exp == 3:
        config = copy.deepcopy(USER_CONFIG_BASE_FINAL)
        trainer = MultiProcessingModulatedSACTrain(
            model_name='base',
            env_class=VariableEnvTimeNoiseBaseFinal,     # Excluded - gaze, e.c., clock noise
            env_setting=config,
            num_cpu=args.cpu
        )
    elif args.exp == 4:
        config = copy.deepcopy(USER_CONFIG_3)
        trainer = MultiProcessingModulatedSACTrain(
            model_name='th',
            env_class=VariableEnvFixedTh,
            env_setting=config,
            num_cpu=args.cpu
        )

    trainer.set_callbacks(
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_ep=args.eval_ep,
    )
    trainer.set_model(
        n_layer_unit=args.un,
        n_layer_depth=args.dn,
        batch_size=args.batch_sz,
        ent=entropy_value,
        concat_layers=cc_layer,
        lr=args.lr * args.cpu,
        gamma=args.gamma,
        target_ent=entropy_target,
        train_freq=args.train_freq
    )

    if args.load != '':
        trainer.load_model(model_name=args.load, ckpt=args.ckpt)

    trainer.run_train(train_steps=2e7)