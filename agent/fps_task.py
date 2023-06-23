"""
FPS Task environment
Class GameState replicates our FPSci experiment (spidershot task)

This code was written by June-Seop Yoon
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from utilities.mymath import *
from utilities.render import draw_game_scene
from utilities.utils import pickle_save
from configs.simulation import *
from configs.experiment import (
    TARGET_RADIUS_SMALL, 
    TARGET_RADIUS_VERY_SMALL, 
    TARGET_RADIUS_LARGE, 
    TARGET_ASPEED_STATIC, 
    TARGET_ASPEED_FAST,
    VALID_INITIAL_GAZE_RADIUS
)


class GameState:
    def __init__(self, sensi=USER_PARAM_MEAN["sensi"]):
        self.ppos = np.zeros(3)     # Player position. (X, Y, Z)
        self.pcam = np.zeros(2)     # Player camera. (Azim, Elev)
        self.tgpos = np.array([1., 0., 0.])     # Target game position. (X, Y, Z)
        self.tmpos = target_pos_monitor(        # Target monitor position. (X, Y)
            self.ppos, self.pcam, self.tgpos
        )
        self.tgspd = 0   # Target in-game orbit speed. unit: deg/sec
        self.toax = perpendicular_vector(   # Target orbit axis (ccw). (Azim, Elev)
            self.tgpos-self.ppos
        )
        self.trad = 4.5e-3                  # Target radius. unit: meter
        self.hpos = np.array([0.0, 0.0])    # Hand position. (X, Y) unit: meter
        self.hvel = np.array([0.0, 0.0])    # Hand velocity. unit: meter/sec
        self.gpos = CROSSHAIR               # Gaze position. unit: meter
        self.sensi = sensi                  # Sensitivity. unit: deg/m.

        self.hrt = 2 * BUMP_INTERVAL
        self.grt = BUMP_INTERVAL

        self.eye_pos = EYE_POSITION

        self.t_traj = []


    def reset(
        self,
        pcam=None,
        tgpos=None,
        tgpos_sp=None,
        tmpos=None,
        eccH=TARGET_SPAWN_BD_HOR,
        eccV=TARGET_SPAWN_BD_VER,
        toax=None,
        tmdir=None,
        session=None,
        tgspd=None,
        trad=None,
        gpos=None,
        hrt=None,
        grt=None,
        eye_pos=None
    ):
        # Player pos and camera
        self.ppos = np.zeros(3)
        if pcam is not None:
            self.pcam = pcam
        else:
            _a = np.random.uniform(-np.pi, np.pi)
            _r = np.random.uniform(0, TARGET_RADIUS_BASE)   # Reference target size
            ref_aim = target_pos_game(
                np.zeros(3), np.zeros(2), 
                np.array([np.cos(_a), np.sin(_a)]) * _r
            )
            self.pcam = ct2sp(*ref_aim)

        # Target setting by monitor coordinate
        if tmpos is not None:
            self.tmpos = tmpos
            self.tgpos = target_pos_game(self.ppos, self.pcam, self.tmpos)
        else:
            # Target setting by game coordinate
            if tgpos is not None: self.tgpos = tgpos
            elif tgpos_sp is not None:
                self.tgpos = sp2ct(*tgpos_sp)
            else:
                # Random spawn
                self.tgpos = sp2ct(
                    np.random.uniform(*eccH) * rand_sign(),
                    np.random.uniform(*eccV) * rand_sign(),
                )
            self.update_tmpos()
        
        # Random orbit axis
        if toax is not None: self.toax = toax
        else:
            self.toax = perpendicular_vector(
                self.tgpos - self.ppos, a=tmdir
            )   # Azim, Elev

        # Target setting by session
        if session is not None:
            if session[1] == 's': self.trad = TARGET_RADIUS_SMALL
            elif session[1] == 'l': self.trad = TARGET_RADIUS_LARGE
            elif session[1] == 'v': self.trad = TARGET_RADIUS_VERY_SMALL

            if session[0] == 's': self.tgspd = TARGET_ASPEED_STATIC
            elif session[0] == 'f': self.tgspd = TARGET_ASPEED_FAST

        else:
            if trad is None and tgspd is None:
                self.trad, self.tgspd = self._difficulty_sampler_gaussian()
            else:
                if trad is not None:
                    self.trad = trad
                else:
                    self.trad = np.random.uniform(
                        TARGET_RADIUS_MIN, TARGET_RADIUS_MAX
                    )
                if tgspd is not None:
                    self.tgspd = tgspd
                else:
                    self.tgspd = np.random.uniform(0, self.__max_speed_by_radius(self.trad))
            
        # Hand, gaze
        self.hpos = np.zeros(2)
        self.hvel = np.zeros(2)

        if gpos is not None:
            self.gpos = gpos
        else:
            while True:
                gpos = np.random.normal(0, 6.19569e-3, size=2)
                if np.linalg.norm(gpos - CROSSHAIR) < VALID_INITIAL_GAZE_RADIUS:
                    self.gpos = gpos
                    break

        # Reaction time - distribution of experiments
        if hrt is not None:
            self.hrt = np.clip(hrt, BUMP_INTERVAL, 3*BUMP_INTERVAL)
        else:
            while True:
                self.hrt = np.random.normal(0.154, 0.028)
                if BUMP_INTERVAL <= self.hrt <= 3*BUMP_INTERVAL: break
        if grt is not None:
            self.grt = np.clip(grt, 0.05, 0.3)
        else:
            while True:
                self.grt = np.random.normal(0.122, 0.045)
                if 0.05 <= self.grt <= 0.3: break

        # Eye position
        if eye_pos is not None:
            self.eye_pos = eye_pos
        else:
            self.eye_pos = EYE_POSITION + np.clip(
                np.random.normal(
                    np.zeros(3),
                    EYE_POSITION_STD
                ),
                a_min=-EYE_POSITION_BOUND,
                a_max=EYE_POSITION_BOUND
            )

        return dict(
            pcam = self.pcam,
            tgpos = self.tgpos,
            tmpos = self.tmpos,
            toax = self.toax,
            tgspd = self.tgspd,
            trad = self.trad,
            gpos = self.gpos,
            hrt = self.hrt,
            grt = self.grt,
            eye_pos = self.eye_pos,
            session = session,
        )


    def _difficulty_sampler_gaussian(
        self, 
        sigma_r=0.004,
        min_s_p=0.035,
    ):
        """New target condition sampler. All region with same probability"""
        r = -1.0
        while r > TARGET_RADIUS_MAX or r < TARGET_RADIUS_MIN:
            r = np.random.normal(TARGET_RADIUS_BASE, sigma_r)
        if np.random.uniform(0, 1) < min_s_p:
            s = 0
        else:
            s = np.random.uniform(0, 1) * self.__max_speed_by_radius(r)
        
        return r, s
    
    def __max_speed_by_radius(self, r):
        # Linearly decrease max. speed from baseline radius to min radius
        if r > TARGET_RADIUS_BASE: return TARGET_ASPEED_MAX
        else:
            return TARGET_ASPEED_MAX * (
                r - TARGET_RADIUS_MIN
            ) / (
                TARGET_RADIUS_BASE - TARGET_RADIUS_MIN
            )


    def update_tmpos(self, record=False):
        self.tmpos = target_pos_monitor(self.ppos, self.pcam, self.tgpos)
        if record: self.t_traj.append(self.tmpos)

    
    def fixate(self, gp):
        self.gpos = gp


    def orbit(self, dt, record=False):
        '''Update target (dt: sec)'''
        self.tgpos = rot_around(
            self.tgpos, 
            self.tgspd * dt, 
            *self.toax
        )
        self.update_tmpos(record=record)

    
    def move_hand(self, dp, v=None, record=False):
        self.hpos += dp
        # self.pcam += dp * self.sensi
        self.pcam = self.hpos * self.sensi
        if v is not None: self.hvel = v
        self.update_tmpos(record=record)
    

    def set_hand_pos(self, p):
        self.hpos = p
        self.pcam = self.hpos * self.sensi
        self.update_tmpos(record=False)

    
    def tvel_by_orbit(self, dt=1):
        '''Return speed on monitor by orbit'''
        tm_0 = self.tmpos
        self.orbit(dt)
        tm_n = self.tmpos
        self.orbit(-dt)
        return (tm_n - tm_0) / dt     # m/sec
    

    def tvel_by_aim(self, dt=0.01):
        '''Return speed on moniotr by aim'''
        tm_0 = self.tmpos
        self.move_hand(self.hvel*dt)
        tm_n = self.tmpos
        self.move_hand(-self.hvel*dt)
        return (tm_n - tm_0) / dt     # m/sec
    

    def tvel_if_aim(self, mp, interval):
        '''Return mean speed on monitor by given aim plan'''
        tm = [self.tmpos]
        delta_hand = mp[1:] - mp[:-1]
        for dh in delta_hand:
            self.move_hand(dh)
            tm.append(self.tmpos)
        tm = np.array(tm)
        self.move_hand(mp[0] - mp[-1])
        return np.mean((tm[1:] - tm[:-1]) / interval, axis=0)

    
    def tmpos_if_hand_move(self, dp):
        self.move_hand(dp)
        tm_n = self.tmpos
        self.move_hand(-dp)
        return tm_n
    

    def tvel_if_hand_move(self, dp, dt=0.01):
        tm_0 = self.tmpos
        tm_n = self.tmpos_if_hand_move(dp)
        return (tm_n - tm_0) / dt
    

    def cam_if_hand_move(self, dp):
        self.move_hand(dp)
        cam_n = self.pcam
        self.move_hand(-dp)
        return cam_n
    

    def tgspd_if_tvel(self, v, dt=1):
        '''Using current monitor target vel., estimate orbit speed'''
        tc0 = self.tgpos
        tcn = target_pos_game(self.ppos, self.pcam, self.tmpos + v * dt)
        # Angle btw tc0 and tcn
        da = angle_btw(tc0, tcn)
        return da / dt
    

    def tmpos_delta_if_orbit(self, a):
        if self.tgspd == 0 or a == 0: return np.zeros(2)
        tm_0 = self.tmpos
        self.orbit(a / self.tgspd)
        tm_n = self.tmpos
        self.orbit(-a / self.tgspd)
        return tm_n - tm_0
    

    def target_crosshair_distance(self):
        return np.linalg.norm(self.tmpos - CROSSHAIR)
    

    def crosshair_on_target(self):
        d = self.target_crosshair_distance()
        return d < self.trad, d

    
    def target_out_of_monitor(self):
        return np.any(self.tmpos > MONITOR_BOUND) or np.any(self.tmpos < -MONITOR_BOUND)


    def show_monitor(self):
        plt.figure(figsize=(8, 4.5))
        target = plt.Circle(self.tmpos, self.trad, color='r')
        plt.gcf().gca().add_patch(target)
        plt.scatter(*CROSSHAIR, zorder=10, s=3, color='k')
        plt.scatter(*self.gpos, zorder=10, s=3, color='b')
        plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
        plt.xticks(np.linspace(-MONITOR_BOUND[X], MONITOR_BOUND[X], 5))
        plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
        plt.yticks(np.linspace(-MONITOR_BOUND[Y], MONITOR_BOUND[Y], 5))
        plt.grid(True); plt.show(); plt.close()

    
    def draw_scene(self, res=720, draw_gaze=True, gray_target=False):
        return draw_game_scene(
            res,
            self.pcam,
            self.tmpos,
            self.trad,
            self.gpos,
            draw_gaze=draw_gaze,
            gray_target=gray_target
        )

    
    def save_random_trial_condition(self, n):
        conds = [
            self.reset() for _ in range(n)
        ]
        pickle_save(f"{PATH_MAT}task_cond_{n}.pkl", conds)



if __name__ == "__main__":
    g = GameState()
    g.reset(session='fsw', tmdir=180)
    g.show_monitor()
    g.orbit(0.3)
    g.show_monitor()
    # for _ in range(1000):
    #     g.reset(
    #         eccH=[36.999, 37],
    #         eccV=[20.999, 21],
    #         tgspd=TARGET_ASPEED_MAX
    #     )
    #     v.append(np.linalg.norm(g.tvel_by_orbit(dt=0.5)))
    
    # print(np.max(v))

    # for _ in range(1000):
    #     g.reset()
    #     v.append(g.gpos)
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4.5))
    # plt.scatter(*np.array(v).T, s=0.1)
    # plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
    # plt.xticks(np.linspace(-MONITOR_BOUND[X], MONITOR_BOUND[X], 5))
    # plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
    # plt.yticks(np.linspace(-MONITOR_BOUND[Y], MONITOR_BOUND[Y], 5))
    # plt.show()


    # g.reset(
    #     tmpos=np.array([-0.13013, 0.066928]),
    #     toax=np.array([44.127, -44.916]),
    #     session='fsw'
    # )
    # x = []
    # for t in np.linspace(0, 0.6, 10):
    #     g.orbit(t)
    #     x.append(g.tmpos)
    # plt.figure(figsize=(8, 4.5))
    # for xx in x:
    #     target = plt.Circle(xx, g.trad, color='r')
    #     plt.gcf().gca().add_patch(target)
    # plt.scatter(*CROSSHAIR, zorder=10, s=2, color='k')
    # plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
    # plt.xticks(np.linspace(-MONITOR_BOUND[X], MONITOR_BOUND[X], 5))
    # plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
    # plt.yticks(np.linspace(-MONITOR_BOUND[Y], MONITOR_BOUND[Y], 5))
    # plt.grid(True); plt.show(); plt.close()

    pass