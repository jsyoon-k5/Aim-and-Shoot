import numpy as np
import matplotlib.pyplot as plt

import sys, os

sys.path.append("..")

from configs.experiment import *
from expdata.ans_experiment import ExperimentData
from expdata.data_process import load_experiment
from utilities.mymath import *

d = ExperimentData('2also0', 'custom')
# lot = load_experiment(player='2also0', mode='custom', session_id)
d.trial_hand_metrics('ssw', 0, 21, plot=True)
# d.export_to_csv()

# ts = []
# hs = []

# for sn in SES_NAME_ABBR:
#     for bn in range(2):
#         for tn in range(TRIAL_NUM_PER_BLOCK):
#             if d.check_trial_exist(sn, bn, tn):
#                 _ts, _hs = d.trial_hand_speed_profile(sn, bn, tn, plot=False)

#                 ts.append(_ts)
#                 hs.append(_hs)

# ts = np.concatenate(ts)
# hs = np.concatenate(hs)