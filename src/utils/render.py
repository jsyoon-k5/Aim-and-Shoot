import cv2
import numpy as np
from box import Box

from ..config.config import ENV
from ..config.constant import VIDEO
from ..utils.mymath import Convert

R_WIDTH = ENV.window_ratio_width
R_HEIGHT = ENV.window_ratio_height


# R, G, B
COLOR = Box(dict(
    RED = (255, 0, 0),
    GREEN = (0, 255, 0),
    BLUE = (0, 0, 255),
    BLACK = (0, 0, 0),
    WHITE = (255, 255, 255),
    GRAY = (40, 40, 40),
    DARKGRAY = (20, 20, 20),
    SKYBLUE = (0, 176, 240),
    VIOLET = (232, 24, 163),
))


class RENDER:
    ASPECT_RATIO = R_HEIGHT / R_WIDTH

    def img_resolution(resolution):
        return (resolution//R_HEIGHT*R_WIDTH, resolution)


    def convert_meter_to_pixel(point, resolution, monitor_qt):
        # monitor_qt := [W, H] / 2
        # point \in [-monitor_qt, monitor_qt]
        pix = np.rint((point + monitor_qt) / (monitor_qt * 2) * np.array(RENDER.img_resolution(resolution))).astype(int)
        return pix


    def empty_scene(resolution, color):
        w, h = RENDER.img_resolution(resolution)
        return (np.ones((h, w, 3)) * color).astype(np.uint8)


    def draw_circle(scene, center, radius, color, thick=-1):
        r, g, b = color
        return cv2.circle(scene, center, radius, (b, g, r), thick)
    

    def draw_line(scene, p1, p2, color, thick=1):
        r, g, b = color
        return cv2.line(
            scene, p1, p2, (b, g, r), thick
        )


    def put_text_at(scene, msg, pos, scale, color):
        r, g, b = color
        img_height, img_width = scene.shape[:2]
        # abs_pos = (int(pos[0] * img_width), int(pos[1] * img_height))
        return cv2.putText(
            scene, 
            str(msg), 
            pos,
            VIDEO.FONT,
            scale, 
            (b, g, r), 
            1, 
            cv2.LINE_AA
        )


    def convert_htraj_to_pixel(htraj_p, resolution, expand_rate=1.2, size_ratio=0.35):
        min_htj = np.min(htraj_p, axis=0)
        translated_traj = htraj_p - min_htj
        
        meter_range = np.max(translated_traj) * expand_rate
        
        # Center the trajectory within the meter range
        center_translation = np.max(translated_traj, axis=0) / 2 - np.array([meter_range] * 2) / 2
        centered_htj = translated_traj - center_translation
        
        # Convert meter range to pixel range and calculate pixel coordinates
        pixel_range = int(resolution * size_ratio)
        htj_pixel = (centered_htj * pixel_range / meter_range).astype(int)
        
        return htj_pixel, meter_range, pixel_range


    def draw_trajectory(scene, points, color):
        print(points)
        if len(points) >= 2:
            for i in range(len(points)-1):
                scene = cv2.line(
                    scene, 
                    (points[i][0], scene.shape[0] - points[i][1]), 
                    (points[i+1][0], scene.shape[0] - points[i+1][1]), 
                    color, 1
                )
        return scene


    def draw_messages(scene, msg_list, s=0.5, skip_space=0):
        scale_ratio = scene.shape[0] / 720
        for i, [msg, color] in enumerate(msg_list):
            scene = RENDER.put_text_at(
                scene, msg,
                (int(10 * scale_ratio), int(25 * (2*s) * scale_ratio) * (i+1+skip_space)),
                s * scale_ratio, color
            )
        return scene




BACKGROUND_REFERENCE = [
    Convert.sphr2cart(az, el) for az in np.linspace(0, 360, 16)[:-1] for el in np.linspace(-80, 80, 16)
]