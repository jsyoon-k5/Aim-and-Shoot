"""
FPS Task environment
Class GameState replicates our FPSci experiment (spidershot task)

This code was written by June-Seop Yoon
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from utils.mymath import *
from utils.render import draw_game_scene
from utils.utils import pickle_save, pickle_load
from configs.common import *
from configs.simulation import *
from configs.experiment import SES_NAME_ABBR

DELTA_TIME = 1e-6


class GameState:
    def __init__(self, sensi=1000):
        self.ppos = np.zeros(3)     # Player position. (X, Y, Z)
        self.pcam = np.zeros(2)     # Player camera. (Azim, Elev)
        self.tgpos = np.array([1., 0., 0.])     # Target game position. (X, Y, Z)
        self.tmpos = target_monitor_position(        # Target monitor position. (X, Y)
            self.ppos, self.pcam, self.tgpos
        )
        self.tgspd = 0   # Target in-game orbit speed. unit: deg/sec
        self.toax = perpendicular_vector(   # Target orbit axis (ccw). (Azim, Elev)
            self.tgpos-self.ppos
        )
        self.trad = 4.5e-3                  # Target radius. unit: m
        self.hpos = np.array([0.0, 0.0])    # Hand position. (X, Y) unit: m
        self.hvel = np.array([0.0, 0.0])    # Hand velocity. unit: m/s
        self.gpos = CROSSHAIR               # Gaze position. unit: m
        self.sensi = sensi                  # Sensitivity. unit: deg/m.

        self.hrt = 0.2
        self.grt = 0.1

        self.head_pos = HEAD_POSITION

        self.t_traj = []


    def reset(
        self,
        pcam=None,  # Az, El
        tgpos=None, # X, Y, Z
        tgpos_sp=None,  # Az, El
        tmpos=None, # X, Y
        eccH=TARGET_SPAWN_BD_HOR,
        eccV=TARGET_SPAWN_BD_VER,
        toax=None,  # Az, El
        tmdir=None, # scalar
        session=None,
        fix_to_experiment_session=False,
        follow_exp_prior=False,
        tgspd=None, # deg/s
        trad=None,
        sample_easy_only=True,
        gpos=None,
        hrt=None,
        grt=None,
        head_pos=None,
        boundary_p=0.05
    ):
        # Player pos and camera
        self.ppos = np.zeros(3)
        if pcam is not None:
            self.pcam = pcam
        else:
            _a = np.random.uniform(-np.pi, np.pi)
            _r = np.random.uniform(0, TARGET_RADIUS_BASE)   # Reference target size
            ref_aim = target_game_position(
                np.zeros(3), np.zeros(2), 
                np.array([np.cos(_a), np.sin(_a)]) * _r
            )
            self.pcam = cart2sphr(*ref_aim)

        # Target setting by monitor coordinate
        if tmpos is not None:
            self.tmpos = tmpos
            self.tgpos = target_game_position(self.ppos, self.pcam, self.tmpos)
        else:
            # Target setting by game coordinate
            if tgpos is not None: self.tgpos = tgpos
            elif tgpos_sp is not None:
                self.tgpos = sphr2cart(*tgpos_sp)
            else:
                # Random spawn
                self.tgpos = sphr2cart(
                    np.random.uniform(*eccH) * random_sign(),
                    np.random.uniform(*eccV) * random_sign(),
                )
            self.update_tmpos()
        
        # Random orbit axis
        if toax is not None: self.toax = toax
        else:
            self.toax = perpendicular_vector(
                self.tgpos - self.ppos, a=tmdir
            )   # Azim, Elev

        # Target setting by session
        if session is None:
            if fix_to_experiment_session:
                session = SES_NAME_ABBR[:5][np.random.randint(5)]
            elif follow_exp_prior:
                if np.random.uniform(0, 1) > 0.5:
                    session = None
                else: session = SES_NAME_ABBR[:5][np.random.randint(5)]

        if session is not None:
            # Experiment setting
            if session[1] == 's': self.trad = TARGET_RADIUS_SMALL
            elif session[1] == 'l': self.trad = TARGET_RADIUS_LARGE
            elif session[1] == 'v': self.trad = TARGET_RADIUS_VERY_SMALL

            if session[0] == 's': self.tgspd = TARGET_ASPEED_STATIC
            elif session[0] == 'f': self.tgspd = TARGET_ASPEED_FAST

            # if trad is not None or tgspd is not None:
                # print("Warning: Task condition radius or speed is overwrited by session setting.")
        else:
            if sample_easy_only:
                self.trad, self.tgspd = self._difficulty_sampler(boundary_p=boundary_p)
            else:
                if trad is not None:
                    self.trad = trad
                else:
                    # self.trad = linear_denormalize(np.random.uniform(-(1+boundary_p), 1+boundary_p), TARGET_RADIUS_VERY_SMALL, TARGET_RADIUS_LARGE)
                    self.trad = np.random.uniform(TARGET_RADIUS_MINIMUM, TARGET_RADIUS_MAXIMUM)
                if tgspd is not None:
                    self.tgspd = tgspd
                else:
                    self.tgspd = linear_denormalize(np.random.uniform(-(1+boundary_p), 1), 0, TARGET_ASPEED_MAX)
                
            # self.trad = trad if trad is not None else  np.random.uniform(TARGET_RADIUS_VERY_SMALL, TARGET_RADIUS_LARGE)
            # self.tgspd = tgspd if tgspd is not None else np.random.uniform(0, TARGET_ASPEED_FAST)
            

        # Hand, gaze
        self.hpos = self.pcam / self.sensi
        self.hvel = np.zeros(2)

        if gpos is not None:
            self.gpos = gpos
        else:
            while True:
                gpos = np.random.normal(0, 0.00619569, size=2)
                if np.linalg.norm(gpos - CROSSHAIR) < VALID_INITIAL_GAZE_RADIUS:
                    self.gpos = gpos
                    break

        # Reaction time - distribution of experiments
        if hrt is not None:
            self.hrt = np.clip(hrt, MIN_HAND_REACTION, MAX_HAND_REACTION)
        else:
            while True:
                self.hrt = stats.skewnorm(1.3984, 0.1316, 0.03384).rvs(1)[0]
                if MIN_HAND_REACTION <= self.hrt <= MAX_HAND_REACTION: break

        if grt is not None:
            self.grt = np.clip(grt, MIN_GAZE_REACTION, MAX_GAZE_REACTION)
        else:
            while True:
                self.grt = stats.skewnorm(6.0445, 0.0801, 0.0881).rvs(1)[0]
                if MIN_GAZE_REACTION <= self.grt <= MAX_GAZE_REACTION: break

        # Eye position
        if head_pos is not None:
            self.head_pos = head_pos
        else:
            self.head_pos = HEAD_POSITION + np.clip(
                np.random.normal(
                    np.zeros(3),
                    HEAD_POSITION_STD
                ),
                a_min=-HEAD_POSITION_BOUND,
                a_max=HEAD_POSITION_BOUND
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
            head_pos = self.head_pos,
            session = session,
        )


    def _difficulty_sampler(self, boundary_p=0.01):
        """New target condition sampler. All region with same probability"""
        while True:
            r = np.random.uniform(TARGET_RADIUS_MINIMUM, TARGET_RADIUS_MAXIMUM)
            s = linear_denormalize(np.random.uniform(-(1+boundary_p), 1), TARGET_ASPEED_STATIC, TARGET_ASPEED_MAX)
            if s <= self.__max_speed_by_radius(r): break
        return r, s
    
    def __max_speed_by_radius(self, r):
        # Linearly decrease max. speed from baseline radius to min radius
        if r > TARGET_RADIUS_SMALL: return TARGET_ASPEED_MAX
        else:
            return TARGET_ASPEED_MAX * (r - TARGET_RADIUS_MINIMUM) / (TARGET_RADIUS_SMALL - TARGET_RADIUS_MINIMUM)


    def update_tmpos(self, record=False):
        self.tmpos = target_monitor_position(self.ppos, self.pcam, self.tgpos)
        if record: self.t_traj.append(self.tmpos)

    
    def fixate(self, gp):
        self.gpos = gp


    def orbit(self, dt, record=False):
        '''Update target (dt: s)'''
        self.tgpos = rotate_point_around_axis(
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


    def set_cam_angle(self, a):
        self.pcam = a

    
    def tvel_by_orbit(self, dt=DELTA_TIME):
        '''Return speed on monitor by orbit'''
        tm_0 = self.tmpos
        tm_n = target_monitor_position(
            self.ppos, self.pcam, 
            rotate_point_around_axis(
                target_game_position(self.ppos, self.pcam, self.tmpos),
                self.tgspd * dt, *self.toax
            )
        )
        return (tm_n - tm_0) / dt     # m/sec
    

    def tvel_by_aim(self, dt=DELTA_TIME):
        '''Return speed on moniotr by aim'''
        tm_0 = self.tmpos
        self.move_hand(self.hvel*dt)
        tm_n = self.tmpos
        self.move_hand(-self.hvel*dt)
        return (tm_n - tm_0) / dt     # m/sec
    

    def tvel_if_aim(self, hand_v, dt=DELTA_TIME):
        '''Return mean speed on monitor by given aim plan'''
        pcam_new = self.pcam + hand_v * dt * self.sensi
        tmpos_new = target_monitor_position(self.ppos, pcam_new, self.tgpos)
        return (tmpos_new - self.tmpos) / dt

    
    def toax_if_tmpos(self, tmpos, tgvel, dt=DELTA_TIME):
        tgpos1 = target_game_position(self.ppos, self.pcam, tmpos)
        tgpos2 = target_game_position(self.ppos, self.pcam, tmpos + dt * tgvel)
        return cart2sphr(*np.cross(tgpos1, tgpos2))

    
    def tmpos_if_hand_move(self, dp):
        self.move_hand(dp)
        tm_n = self.tmpos
        self.move_hand(-dp)
        return tm_n
    
    def delta_tmpos_if_hand_move(self, tmpos, dp):
        tgpos = target_game_position(self.ppos, self.pcam, tmpos)
        new_pcam = self.pcam + dp * self.sensi
        new_tmpos = target_monitor_position(self.ppos, new_pcam, tgpos)
        return new_tmpos - tmpos
    

    def tmpos_if_hand_and_orbit(self, tmpos, dp, a, toax=None):
        """
        Given virtual tmpos and hand displacement dp, angle a,
        return predicted tmpos
        """
        toax = self.toax if toax is None else toax
        pcam_new = self.pcam + dp * self.sensi
        tgpos0 = target_game_position(self.ppos, self.pcam, tmpos)
        tgpos1 = rotate_point_around_axis(tgpos0, a, *toax)
        return np.clip(target_monitor_position(self.ppos, pcam_new, tgpos1), -2*MONITOR_BOUND, 2*MONITOR_BOUND)


    def ttraj_if_hand_and_orbit(self, hpos, tmpos, interval, tgvel):
        pcam_traj = hpos * self.sensi
        tgpos = target_game_position(self.ppos, pcam_traj[0], tmpos)
        return np.array([
            target_monitor_position(self.ppos, pcam, tgpos) + i * interval * tgvel for i, pcam in enumerate(pcam_traj)
        ])

        # tgpos_traj = [target_game_position(self.ppos, pcam_traj[0], tmpos)]
        # for i in range(1, len(hpos)):
            # tgpos_traj.append(rotate_point_around_axis(tgpos_traj[0], i * a * interval, *toax))
        
        # return np.array([target_monitor_position(self.ppos, pcam, tgpos) for pcam, tgpos in zip(pcam_traj, tgpos_traj)])


    def tvel_if_hand_and_orbit(self, tmpos, hand_vel, dt=DELTA_TIME):
        pcam_new = self.pcam + hand_vel * dt * self.sensi
        tgpos0 = target_game_position(self.ppos, self.pcam, tmpos)
        tgpos1 = rotate_point_around_axis(tgpos0, self.tgspd * dt, *self.toax)
        tmpos_new = target_monitor_position(self.ppos, pcam_new, tgpos1)
        return (tmpos_new - tmpos) / dt
    

    def tvel_if_hand_move(self, dp, dt=DELTA_TIME):
        tm_0 = self.tmpos
        tm_n = self.tmpos_if_hand_move(dp)
        return (tm_n - tm_0) / dt
    

    def cam_if_hand_move(self, dp):
        return self.pcam + dp * self.sensi
    

    def tgspd_if_tvel(self, tmpos, tvel, dt=DELTA_TIME):
        '''Using current monitor target vel., estimate orbit speed'''
        tc0 = target_game_position(self.ppos, self.pcam, tmpos)
        tcn = target_game_position(self.ppos, self.pcam, tmpos + tvel * dt)
        # Angle btw tc0 and tcn
        da = angle_between(tc0, tcn)
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

    
    def sample_task_condition(self, n, **cond):
        return [self.reset(**cond) for _ in range(n)]

    
    def save_random_trial_condition(self, n):
        conds = [
            self.reset() for _ in range(n)
        ]
        pickle_save(f"{PATH_MAT}task_cond_{n}.pkl", conds)

    
    # def diff_index_int(self):
    #     dist_idx = int(np.linalg.norm(self.tmpos) > 0.0905)
    #     if self.tgspd < TARGET_ASPEED_MAX / 2:
    #         if self.trad < TARGET_RADIUS_BASE: return 2 + 5*dist_idx
    #         elif self.trad < (TARGET_RADIUS_BASE + TARGET_RADIUS_MAX) / 2: return 1 + 5*dist_idx
    #         else: return 5*dist_idx
    #     else:
    #         if self.trad < (TARGET_RADIUS_BASE + TARGET_RADIUS_MAX) / 2: return 4 + 5*dist_idx
    #         else: return 3 + 5*dist_idx

    
    # def save_specified_diff_trial_condition(self, n, idx=0):
    #     assert idx in list(range(10))
    #     conds = []
    #     _n = 0
    #     while True:
    #         cond = self.reset()
    #         if self.diff_index_int() == idx:
    #             conds.append(cond)
    #             _n += 1
    #         if _n == n: break
    #     pickle_save(f"{PATH_MAT}task_cond_idx{idx}_{n}.pkl", conds)




if __name__ == "__main__":
    g = GameState()
    
    r = []
    for _ in range(10000):
        c = g.reset()
        t = np.log2(np.linalg.norm(c["tmpos"]) / (2*c["trad"]) + 1) / 6
        s = c["tgspd"] / TARGET_ASPEED_MAX
        r.append(t * (1+s))
    
    plt.hist(r, bins=50)
    plt.show()

    print(max(r), min(r))

    # d = []
    # se = []
    # for _ in range(10000):
    #     c = g.reset(session='fsw', tmpos=np.random.uniform(-MONITOR_BOUND * 0.9, MONITOR_BOUND * 0.9))
    #     d.append(np.linalg.norm(c["tmpos"]))
    #     se.append(g.tgspd - g.tgspd_if_tvel(g.tmpos, g.tvel_by_orbit()))
    
    # plt.scatter(d, se, s=0.1)
    # plt.show()

    # d = []
    # se = []
    # for _ in range(10000):
    #     c = g.reset(session='fsw', toax=toax, tmpos=np.random.uniform(-MONITOR_BOUND * 0.9, MONITOR_BOUND * 0.9))
    #     d.append(np.linalg.norm(c["tmpos"]))
    #     se.append(np.linalg.norm(c["toax"] - g.toax_if_tmpos(g.tmpos, g.tvel_by_orbit())))
    
    # plt.scatter(d, se, s=0.1)
    # plt.show()

    # r = []
    # s = []
    # for _ in range(10000):
    #     c = g.reset()
    #     r.append(c["trad"])
    #     s.append(c["tgspd"])
    
    # plt.scatter(r, s, s=0.1)
    # plt.show()
    # x = g.tmpos_if_hand_and_orbit(
    #     np.zeros(2),
    #     np.zeros(2),
    #     np.zeros(2)
    # )

    # from agent.module_perception import *

    # hand_v = np.array([0, 1.5])
    # tvel_aim = g.tvel_if_aim(hand_v)
    # tvel_sum = g.tvel_if_hand_and_orbit(g.tmpos, hand_v)
    # tvel_orbit = g.tvel_by_orbit()
    # # print(tvel_orbit, tvel_sum - tvel_aim)
    # p1 = np.copy(g.tmpos)
    # g.orbit(0.01)
    # p2 = np.copy(g.tmpos)
    # p2_hat = position_perception(p2, CROSSHAIR, 0.1)
    # err = p2_hat - p2

    # print(g.toax)
    # print(g.toax_if_tmpos(p1, p1 + 0.001 * (tvel_sum - tvel_aim)))

    # x = speed_perception(
    #     tvel=np.ones(2),
    #     tpos=np.zeros(2),
    #     gpos=np.zeros(2),
    #     theta_s=1,
    #     theta_p=1,
    #     theta_f=1
    # )
    # print(x)

    # p = []
    # for _ in range(5000):
    #     c = g.reset(
    #         # tgpos=sphr2cart(1.5/2, 0),
    #         # toax = np.array([0, 90]),
    #         # tgspd=15,
    #     )
    #     p.append(c["head_pos"])
    # p = np.array(p)
    # plt.scatter(*p.T, s=0.1, color='k')
    # plt.hist(p[:,0], bins=100)
    # plt.scatter(r, v, s=0.1)


    # plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
    # plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
    # plt.show()
    
    # # print(p[:,0])
    # # print(np.mean(np.diff(p[:,0]) / 0.01))
    # print((p[0] - p[-1]))

    pass