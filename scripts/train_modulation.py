import sys, os
sys.path.append("..")

from learning.ans_sac import ModulatedSACTrain
from utilities.utils import now_to_string

import argparse

parser = argparse.ArgumentParser(description='Individual training option')
parser.add_argument('--param_space', type=str, default='cog')
# parser.add_argument('--sname', type=str, default='default')

args = parser.parse_args()

trainer = ModulatedSACTrain(
    model_name = f"{args.param_space}",
    modul_space = args.param_space,
    variable_std = None
)
trainer.create_paths()
trainer.set_callbacks(
    save_freq=5e5,
)
trainer.set_model(
    n_layer_unit=1024,
    n_layer_depth=3,
    batch_size=1500,
    concat_layers=[0, 1, 2, 3],
    lr=2e-6,
)
trainer.run_train(train_steps=2e7)