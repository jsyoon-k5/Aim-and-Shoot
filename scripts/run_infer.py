import sys
sys.path.append("..")

from configs.experiment import PLAYERS
from inference.infer_ps import ans_infer

import argparse

parser = argparse.ArgumentParser(description='Inference option')
parser.add_argument('--player', type=str, default='KKW')
# parser.add_argument('--playernum', type=int, default=-1)

args = parser.parse_args()
player = args.player
# pn = args.playernum

assert player in PLAYERS


ans_infer('YEC')
ans_infer('KEH')
ans_infer('Yoonjin04')
ans_infer('khanis0')