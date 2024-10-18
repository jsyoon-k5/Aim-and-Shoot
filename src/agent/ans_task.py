"""
FPS Aim-and-Shoot Task environment
Class GameState replicates our FPSci experiment (spidershot task)

This code was written by June-Seop Yoon
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from box import Box

from ..utils.mymath import Convert, Rotate, random_sign, random_sampler, perpendicular_vector
from ..utils.myutils import pickle_load, pickle_save
from ..config.config import ENV
from ..config.constant import AXIS


class AnSGame:
    def __init__(self, config_name='default', config=None):
        self.config = ENV[config_name] if config is None else config

        self.crosshair = np.zeros(2)

        self.window_qt = np.array([
            self.config.window.monitor.width, 
            self.config.window.monitor.height
        ], dtype=np.float32) / 2


        self.camera = Box(dict(
            pos = np.zeros(3),  # X, Y, Z
            dir = np.zeros(2),  # Azim, Elev
            fov = np.array([self.config.window.fov.width, self.config.window.fov.height])
        ))

        self.hand = Box(dict(
            pos = np.zeros(2),  # X, Y
            vel = np.zeros(2),  # X, Y
            sensi = self.config.mouse_sensi * 1000    # deg/mm -> deg/m
        ))

        self.target = Box(dict(
            rad = 0,
            spd = 0,
            orbit = np.zeros(2),    # Azim, Elev
            pos = dict(
                game = np.zeros(3),     # X, Y, Z
                monitor = np.zeros(2)   # X, Y
            )
        ))

        self.sampler = Box(dict(
            camera = dict(
                max_dev_rad = self.config.range.camera.max_dev_rad  # Threshold for camera direction
            ),
            target = dict(
                gpos = dict(
                    range = dict(
                        azim = [self.config.range.target.pos.azim.min, self.config.range.target.pos.azim.max],
                        elev = [self.config.range.target.pos.elev.min, self.config.range.target.pos.elev.max],
                    ),
                    # bd_sample
                ),
                spd = dict(
                    range = [self.config.range.target.aspeed.min, self.config.range.target.aspeed.max],
                    bd_sample = [self.config.range.target.aspeed.bd_sample.min, self.config.range.target.aspeed.bd_sample.max]
                ),
                rad = dict(
                    range = [self.config.range.target.radius.min, self.config.range.target.radius.max],
                    bd_sample = [self.config.range.target.radius.bd_sample.min, self.config.range.target.radius.bd_sample.max]
                )
            )
        ))
    

    def reset(
        self,
        cdir = None,    # Azim, Elev
        tmpos = None,   # X, Y
        tgpos = None,   # X, Y, Z
        torbit = None,  # Azim, Elev
        tmdir = None,   # Scalar (degree)
        tspd = None,    # Scalar (degree/s)
        trad = None,    # Scalar (m),
        **kwargs
    ):
        # Camera reset
        self.camera.pos = np.zeros(3)
        if cdir is not None:
            self.camera.dir = cdir
        else:
            _a = np.random.uniform(-np.pi, np.pi)
            _r = np.random.uniform(0, self.sampler.camera.max_dev_rad)
            camera_on_reference_click = Convert.monitor2game(
                self.camera.pos,
                np.zeros(2),
                np.array([np.cos(_a), np.sin(_a)]) * _r,
                self.camera.fov,
                self.window_qt
            )
            self.camera.dir = Convert.cart2sphr(*camera_on_reference_click)
        

        ### Target reset
        # Target position (game & monitor)
        if tmpos is not None:   # Monitor coordinate specified
            self.target.pos.monitor = tmpos
            self.target.pos.game = Convert.monitor2game(
                self.camera.pos, self.camera.dir, self.target.pos.monitor, self.camera.fov, self.window_qt
            )
        elif tgpos is not None: # Game coordinate specified (tmpos has higher priority)
            self.target.pos.game = tgpos
            self.target.pos.monitor = self.current_target_monitor_pos()
        else:   # Random sampling
            self.target.pos.game = Convert.sphr2cart(
                az = np.random.uniform(*self.sampler.target.gpos.range.azim) * random_sign(),
                el = np.random.uniform(*self.sampler.target.gpos.range.elev) * random_sign(),
            )
            self.target.pos.monitor = self.current_target_monitor_pos(cdir=np.zeros(2))
        
        # Target radius & speed
        self.target.rad = random_sampler(*self.sampler.target.rad.range, *self.sampler.target.rad.bd_sample) if trad is None else trad
        self.target.spd = random_sampler(*self.sampler.target.spd.range, *self.sampler.target.spd.bd_sample) if tspd is None else tspd

        # Target orbit axis
        self.target.orbit = perpendicular_vector(self.target.pos.game - self.camera.pos, angle=tmdir) if torbit is None else torbit


        # Hand reset
        self.hand.pos = self.camera.dir / self.hand.sensi
        self.hand.vel = np.zeros(2)

        # Note - reaction times and gaze are moved to "agent" condition.
        # These will not be handled in the task class.

        # Target direction
        tdir = Convert.orbit2direction(
            self.camera.pos,
            self.camera.dir,
            self.target.pos.game,
            self.target.orbit,
            self.camera.fov,
            self.window_qt
        )

        return Box(dict(
            cdir = self.camera.dir,
            tmpos = self.target.pos.monitor,
            tgpos = self.target.pos.game,
            torbit = self.target.orbit,
            tspd = self.target.spd,
            trad = self.target.rad,
            tdir = tdir
        ))
    

    def current_target_monitor_pos(self, cpos=None, cdir=None, tgpos=None):
        return Convert.game2monitor(
            self.camera.pos if cpos is None else cpos, 
            self.camera.dir if cdir is None else cdir, 
            self.target.pos.game if tgpos is None else tgpos, 
            self.camera.fov, 
            self.window_qt
        )
    

    def orbit_target(self, dt, inplace=False, unit='s'):
        new_tgpos = Rotate.point_about_axis(
            self.target.pos.game, 
            self.target.spd * dt if unit=='s' else self.target.spd * dt / 1000,
            *self.target.orbit,
            angle_is_degree=True
        )
        new_tmpos = self.current_target_monitor_pos(tgpos=new_tgpos)

        if inplace:
            self.target.pos.game = new_tgpos
            self.target.pos.monitor = new_tmpos
        else:
            return (new_tgpos, new_tmpos)


    def move_hand(self, hand_pos_delta, hand_end_vel):
        self.hand.pos += hand_pos_delta
        self.hand.vel = hand_end_vel
        self.camera.dir = self.hand.pos * self.hand.sensi
        self.orbit_target(dt=0, inplace=True)

    
    def get_current_state(self):
        return dict(
            target_pos_monitor = self.target.pos.monitor.copy(),
            target_pos_game = self.target.pos.game.copy(),
            target_vel_orbit = self.target_monitor_velocity(),
            target_rad = self.target.rad,
            camera_dir = self.camera.dir.copy(),
            hand_pos = self.hand.pos.copy(),
            hand_vel = self.hand.vel.copy()
        )
    
    def replay_and_save(self,
        htraj_p, htraj_v, interval, unit='s', 
        keys=["target_pos_monitor", "hand_pos", "camera_dir"]
    ):
        state = self.get_current_state()
        log = {key: [state[key]] for key in keys}
        if "hand_pos" in log:
            log["hand_pos"] = [htraj_p[0]]
        if "hand_vel" in log:
            log["hand_vel"] = [htraj_v[0]]

        for i in range(1, len(htraj_p)):
            self.orbit_target(interval, unit=unit, inplace=True)
            self.move_hand(htraj_p[i] - htraj_p[i-1], htraj_v[i])
            state = self.get_current_state()
            for key in keys:
                log[key].append(state[key])
        
        for key in keys:
            log[key] = np.array(log[key])
        
        return log
    

    def target_monitor_position(self, initial_target_mpos=None, hand_displacement=np.zeros(2), orbit_angle=0, clip_ratio=2):
        new_cam_dir = self.camera.dir + hand_displacement * self.hand.sensi
        target_pos_game_0 = self.target.pos.game.copy() if initial_target_mpos is None else self._game_position(tmpos=initial_target_mpos)
        target_pos_game_1 = Rotate.point_about_axis(target_pos_game_0, orbit_angle, *self.target.orbit)
        return np.clip(
            self._monitor_position(cdir=new_cam_dir, tgpos=target_pos_game_1),
            a_min = -clip_ratio * self.window_qt,
            a_max = clip_ratio * self.window_qt
        )
    

    def target_monitor_velocity(self, initial_target_mpos=None, hand_vel=np.zeros(2), target_orbit_spd=None, dt=0.001):
        cdir_new = self.camera.dir + hand_vel * dt * self.hand.sensi
        initial_target_mpos = self.target.pos.monitor.copy() if initial_target_mpos is None else initial_target_mpos
        target_pos_game_0 = self.target.pos.game.copy() if initial_target_mpos is None else self._game_position(tmpos=initial_target_mpos)
        target_pos_game_1 = Rotate.point_about_axis(
            target_pos_game_0, 
            self.target.spd * dt if target_orbit_spd is None else target_orbit_spd * dt, 
            *self.target.orbit
        )
        target_mpos_new = self._monitor_position(cdir=cdir_new, tgpos=target_pos_game_1)
        return (target_mpos_new - initial_target_mpos) / dt


    def target_crosshair_distance(self):
        return np.linalg.norm(self.target.pos.monitor - self.crosshair)
    

    def crosshair_on_target(self):
        dist = self.target_crosshair_distance()
        return dist <= self.target.rad, dist
    

    def target_out_of_monitor(self):
        return np.any(self.target.pos.monitor > self.window_qt) or \
            np.any(self.target.pos.monitor < -self.window_qt)

    

    def sample_task_condition(self, n, **cond):
        return [self.reset(**cond) for _ in range(n)]
    

    def _monitor_position(self, cpos=None, cdir=None, tgpos=None, fov=None, window_qt=None):
        return Convert.game2monitor(
            ppos = cpos if cpos is not None else self.camera.pos,
            pcam = cdir if cdir is not None else self.camera.dir,
            gpos = tgpos if tgpos is not None else self.target.pos.game,
            fov = fov if fov is not None else self.camera.fov,
            monitor_qt = window_qt if window_qt is not None else self.window_qt
        )
    
    def _game_position(self, cpos=None, cdir=None, tmpos=None, fov=None, window_qt=None):
        return Convert.monitor2game(
            ppos = cpos if cpos is not None else self.camera.pos,
            pcam = cdir if cdir is not None else self.camera.dir,
            mpos = tmpos if tmpos is not None else self.target.pos.monitor,
            fov = fov if fov is not None else self.camera.fov,
            monitor_qt = window_qt if window_qt is not None else self.window_qt
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    g = AnSGame()
    # s = np.array([Convert.game2monitor(np.zeros(3), np.zeros(2), Convert.sphr2cart(*g.reset().cdir), g.camera.fov, g.window_qt) for _ in range(10000)])
    s = np.array([g.reset().tmpos for _ in range(10000)])
    plt.scatter(*s.T, s=0.1)
    plt.xlim(-g.window_qt[0], g.window_qt[0])
    plt.ylim(-g.window_qt[1], g.window_qt[1])
    plt.gca().set_aspect('equal')
    plt.show()