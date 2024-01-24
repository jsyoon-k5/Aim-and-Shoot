import cv2
import numpy as np

import sys
sys.path.append("..")

from utils.mymath import sphr2cart

CODEC = cv2.VideoWriter_fourcc(*'X264')
FONT = cv2.FONT_HERSHEY_SIMPLEX

MP_EXPAND_RATE = 1.2
MP_SIZE_RATIO = 0.35
MP_OFFSET_RATIO = 0.1
TEXT_SPACING = 25

C_WHITE = (255, 255, 255)
C_GRAY = (40, 40, 40)
C_DARKGRAY = (20, 20, 20)
C_RED = (255, 0, 0)
C_GREEN = (0, 255, 0)
C_BLUE = (0, 0, 255)
C_SKYBLUE = (0, 176, 240)
C_VIOLET = (232, 24, 163)
C_BLACK = (0, 0, 0)

BACKGROUND_REFERENCE = [
    sphr2cart(az, el) for az in np.linspace(0, 360, 16)[:-1] for el in np.linspace(-80, 80, 16)
]