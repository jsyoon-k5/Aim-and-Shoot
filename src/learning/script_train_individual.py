import sys, os, copy
sys.path.append("..")

from configs.simulation import *
from agent.agents import *
from learning.sac_base import SACTrain

import argparse

parser = argparse.ArgumentParser(description='Training option')

parser.add_argument('--batch_sz', type=int, default=2048)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--un', type=int, default=1024)

parser.add_argument('--theta_m', type=float, default=None)
parser.add_argument('--theta_p', type=float, default=None)
parser.add_argument('--theta_s', type=float, default=None)
parser.add_argument('--theta_f', type=float, default=None)
parser.add_argument('--theta_c', type=float, default=None)

args = parser.parse_args()

config = copy.deepcopy(USER_CONFIG_1)
if args.theta_m is not None: config["params_mean"]["theta_m"] = args.theta_m
if args.theta_p is not None: config["params_mean"]["theta_p"] = args.theta_p
if args.theta_s is not None: config["params_mean"]["theta_s"] = args.theta_s
if args.theta_f is not None: config["params_mean"]["theta_f"] = args.theta_f
if args.theta_c is not None: config["params_mean"]["theta_c"] = args.theta_c

trainer = SACTrain(
    env_class=EnvDefault,
    env_setting=config
)
trainer.set_callbacks(eval_freq=5e4)
trainer.set_model(
    n_layer_unit=args.un,
    n_layer_depth=3,
    batch_size=args.batch_sz,
    lr=args.lr,
)
trainer.run_train(train_steps=2e7)