import numpy as np
import matplotlib.pyplot as plt

import sys, copy
sys.path.append("..")

from agent.fps_task import GameState
from agent.ans_agent import *

from configs.common import *

x = Env(
    param_scale_w=dict(
        theta_p=0,
        theta_s=0,
        theta_m=0,
        theta_c=0
    ),
    game_setting=dict(session='ssw')
)
x.reset()

g = copy.deepcopy(x._game)
x._game.show_monitor()
x.step([0.5, 0, 0])
x.step([-1, -1, -1])
x.step([-1, -1, -1])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0.5])
# x.step([0.1, 0, 0])
x._game.show_monitor()
p, v, g, t, _ = x.episode_trajectory()