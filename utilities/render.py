import cv2
import numpy as np

import sys
sys.path.append("..")

from configs.render import *
from configs.common import *
from utilities.mymath import target_pos_monitor, img_resolution

def convert_mon_to_pixel(p, res):
    pix = np.rint((p + MONITOR_BOUND) / MONITOR * np.array(img_resolution(res)))
    return pix.astype(int)


def draw_circle_at(scene, p, r, g, b, size, th):
    return cv2.circle(scene, p, size, (b, g, r), th)


def draw_crosshair(scene, r, g, b, l, rad=0.0015):
    res = scene.shape[0]
    tr = int(round(rad / MONITOR[Y] * res))
    scene = draw_circle_at(scene, (res//9*8, res//2), *C_GREEN, tr, -1)
    scene = draw_circle_at(scene, (res//9*8, res//2), *C_BLACK, tr, 1)
    # scene = cv2.line(scene, (res//9*8, res//2+l), (res//9*8, res//2-l), (b, g, r), 1)
    # scene = cv2.line(scene, (res//9*8+l, res//2), (res//9*8-l, res//2), (b, g, r), 1)
    return scene


def put_text_at(scene, msg, org, s, r, g, b):
    return cv2.putText(
        scene, 
        str(msg), 
        org,
        FONT,
        s, (b, g, r), 1, cv2.LINE_AA
    )


def convert_hand_traj_meter_to_pixel(htj, res):
    """Convert hand plan to pixel unit for rendering"""
    translated_htj = htj - np.min(htj, axis=0)
    mp_range_meter = np.max(translated_htj) * MP_EXPAND_RATE
    trans_to_center = np.max(translated_htj, axis=0) / 2 - np.array([mp_range_meter] * 2) / 2
    mp_range_pixel = int(res * MP_SIZE_RATIO)
    htj_pixel = ((translated_htj - trans_to_center) * mp_range_pixel / mp_range_meter).astype(int)
    return htj_pixel, mp_range_meter, mp_range_pixel


def black_scene(res):
    return np.zeros((res, res//9*16, 3), np.uint8)


def draw_game_scene(
    res,
    pcam,
    tmpos,
    trad,
    gpos,
    draw_gaze=True,
    gray_target=False,
    std_res=720
):
    scene = black_scene(res)
    scale_ratio = res / std_res

    # Draw background objects
    bg_targets = np.array([
        target_pos_monitor(np.zeros(3), pcam, bgt) for bgt in BACKGROUND_REFERENCE
    ])
    bg_targets = convert_mon_to_pixel(bg_targets, res)
    for bgt in bg_targets:
        if (bgt <= np.array(img_resolution(res))).all() and (bgt >= np.zeros(2)).all():
                scene = draw_circle_at(scene, bgt, *C_GRAY, int(3*scale_ratio), -1)

    # Draw target
    tm = convert_mon_to_pixel(tmpos, res)
    tr = int(round(trad / MONITOR[Y] * res))
    if gray_target: scene = draw_circle_at(scene, tm, *C_GRAY, tr, -1)
    else: scene = draw_circle_at(scene, tm, *C_WHITE, tr, -1)

    # Crosshair
    scene = draw_crosshair(scene, *C_GREEN, int(8*scale_ratio))

    # Gaze
    if draw_gaze:
        gp = convert_mon_to_pixel(gpos, res)
        scene = draw_circle_at(scene, gp, *C_RED, int(5*scale_ratio), 2)

    # Flip horizontally (opencv uses flipped starting position in Y-axis)
    scene = cv2.flip(scene, 0)

    return scene


def draw_mouse_trajectory(scene, mp_size_pixel, mp_size_meter, htj, std_res=720):
    """Hand trajectory should be sliced properly"""
    res = scene.shape[0]
    scale_ratio = res / std_res
    ofs = int(mp_size_pixel * MP_OFFSET_RATIO)
    pad_area = scene[res-ofs-mp_size_pixel:res-ofs, ofs:ofs+mp_size_pixel]
    pad_opac = np.ones(pad_area.shape, dtype=np.uint8) * 255
    pad_area = cv2.addWeighted(pad_area, 0.7, pad_opac, 0.3, 0)
    scene[res-ofs-mp_size_pixel:res-ofs, ofs:ofs+mp_size_pixel] = pad_area

    # Trajectory
    x = htj[:,0] + ofs
    y = res - (htj[:,1] + ofs)
    if len(htj) >= 2:
        for i in range(len(htj)-1):
            scene = cv2.line(scene, (x[i], y[i]), (x[i+1], y[i+1]), (0, 255, 255), 1)
    scene = cv2.circle(scene, (x[-1], y[-1]), int(3*scale_ratio), (0, 255, 255), 1)

    # Text
    scene = put_text_at(
        scene, 
        f"Mouse Trajectory (Box scale: {mp_size_meter*100:2.2f}cm)", 
        (ofs, res - (mp_size_pixel + ofs + int(10 * scale_ratio))), 
        0.4 * scale_ratio, *C_WHITE
    )
    return scene


def draw_messages(scene, msg_list, s=0.5, skip_row=0, std_res=720):
    scale_ratio = scene.shape[0] / std_res
    for i, [msg, color] in enumerate(msg_list):
        scene = put_text_at(
            scene, msg,
            (int(10 * scale_ratio), int(TEXT_SPACING * (2*s) * scale_ratio) * (i+1+skip_row)),
            s * scale_ratio, *color
        )
    return scene