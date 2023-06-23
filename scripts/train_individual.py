import sys, os
sys.path.append("..")

from learning.ans_sac import SACTrain
from utilities.utils import now_to_string
from configs.simulation import FIXED_Z_PRESET, COG_SPACE

import argparse

parser = argparse.ArgumentParser(description='Individual training option')
parser.add_argument('--preset_num', type=int, default=0)
parser.add_argument('--sname', type=str, default='default')

args = parser.parse_args()
# preset_num = args.preset_num

# assert preset_num >= 0 and preset_num < 8

# pz = {COG_SPACE[i]:FIXED_Z_PRESET[preset_num][i] for i in range(len(COG_SPACE))}

trainer = SACTrain(
    model_name=f"ind_{args.sname}",
    param_scale_z=dict(
        theta_s=-1,
        theta_p=-1,
        theta_m=-1
    )
)
trainer.create_paths()
trainer.set_callbacks(
    save_freq=5e5,
    eval_ep=256,
)
trainer.set_model(
    n_layer_unit=1024,
    n_layer_depth=3,
    batch_size=1024,
    lr=1e-5,
)
trainer.run_train(train_steps=1e7)