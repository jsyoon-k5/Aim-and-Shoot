import pickle

import sys
sys.path.append("..")

from configs.path import *
from utilities.utils import pickle_load, pickle_save
from expdata.organize_exp_data import player_distance_info

import matplotlib.pyplot as plt

player = 'Yoonjin04'
mode = 'default'
color = 'white'

filename = "traj_Yoonjin04_230508_153212_77"

filepath = f"{PATH_INFER_RESULT % (mode, color)}/{filename}.pkl"

_, (_, _, _, s_ts, s_tx, s_gx) = pickle_load(filepath)
p_ts, p_tx, p_gx = player_distance_info(player, mode, color)

xub = 0.6
yub = 0.1
xticks = [0.05 * i for i in range(13)], [50 * i for i in range(13)]
yticks = [0, 0.025, 0.05, 0.075, 0.1], [0, 2.5, 5, 7.5, 10]

plt.figure(figsize=(6, 1.8))
plt.plot(p_ts, p_tx, linewidth=1, linestyle='-', color='k', label='Target (Player)', zorder=30)
plt.plot(p_ts, p_gx, linewidth=1, linestyle='--', color='k', label='Gaze (Player).',zorder=30)
# plt.plot(s_ts, s_tx, linewidth=1, linestyle='-', color='r', label='Target (Simul)', zorder=30)
# plt.plot(s_ts, s_gx, linewidth=1, linestyle='--', color='r', label='Gaze (Simul)',zorder=30)
plt.legend()
plt.xlabel("Time (millisec)")
plt.ylabel("Distance (cm)")
plt.xlim(0, xub)
plt.xticks(*xticks)
plt.ylim(0, yub)
plt.yticks(*yticks)
plt.grid(True)
plt.show()