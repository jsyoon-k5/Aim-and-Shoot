import numpy as np
import pandas as pd

from datetime import datetime, timezone

import sys, glob
sys.path.append("..")

from configs.experiment import *
from configs.common import *
from configs.path import *

def time_str_to_unix_microsec(timeStr):
    """Convert single time string into UNIX microsecond"""
    ts = datetime.strptime(timeStr, TIME_STR_FORMAT)
    ts = ts.replace(tzinfo=timezone.utc).timestamp()
    return int(ts * MM)


def time_list_to_unix_microsec(timeList):
    """Convert time string list into UNIX microsecond"""
    return list(map(time_str_to_unix_microsec, timeList))


def queryDB(db, command, args=(), convert_time=True):
    """SQL command query"""
    cur = db.cursor()
    q = cur.execute(command, args)
    cols = [column[0] for column in q.description]
    _d = pd.DataFrame.from_records(data=q.fetchall(), columns=cols)
    # Convert time to UNIX microsecond
    if convert_time:
        for col in cols:
            if 'time' in col:
                try:
                    _d[col] = pd.Series(time_list_to_unix_microsec(_d[col]))
                except: continue
    return _d


def load_gazepoint_data(username):
    return (
        pd.read_csv(PATH_DATA_EXP_RAW_GAZE % (username, username)),
        pd.read_csv(PATH_DATA_EXP_RAW_CALIB % (username, username))
    )


def load_mouse_data(username):
    return pd.read_csv(PATH_DATA_EXP_RAW_MOUSE % (username, username))


def data_in_timerange(df, timecol, st, et):
    """Load df where st < timecol < et"""
    return df[(df[timecol] >= st) & (df[timecol] <= et)]


def trial_index_from_name(name):
    try: return int(name.split('_')[2])
    except: return np.nan


def trial_index_from_names(names):
    return list(map(trial_index_from_name, names))


def sess_id_from_name(name):
    try: 
        n = name.split("_")[0]
        if n == "calib": n = "verif"
        return n
    except: return ''
    

def sess_id_from_names(names):
    return list(map(sess_id_from_name, names))


def trad_from_sess(sn):
    if sn[1] == 's': return TARGET_RADIUS_SMALL
    elif sn[1] == 'l': return TARGET_RADIUS_LARGE
    elif sn[1] == 'v': return TARGET_RADIUS_VERY_SMALL
    else:
        print("Invalid session name ...")
        return None
    

def tgspd_from_sess(sn):
    if sn[0] == 's': return TARGET_ASPEED_STATIC
    elif sn[0] == 'f': return TARGET_ASPEED_FAST
    else:
        print("Invalid session name ...")
        return None


def tcolor_from_sess(sn):
    if sn[-1] == 'w': return COLOR[0]
    elif sn[-1] == 'g': return COLOR[1]
    else:
        print("Invalid session name ...")
        return None
    

def parse_tier_by_playername(player):
    for tier in TIER:
        if player in PLAYER[tier]:
            return tier
    print("Invalid session name ...")
    return None


def ses_name_abbr_to_full(sn):
    return SES_NAME_FULL[SES_NAME_ABBR.index(sn)]


### GRAVEYARD: DEPRECATED
def load_raw_gazepoint_data(username):
    """Return merged pandas Dataframe gazepoint data with calibration result"""
    fn = glob.glob(PATH_DATA_EXP_RAW_GAZE % username)
    fn.sort()

    # Calibration result
    with open(fn[0], 'r') as f:
        calib_result_raw = f.readline().replace('"', '').split()[1:-1]
        calib_result = {val:[] for val in F_CALIB}
        for calibPt in range(1, N_CALIB_PT + 1):
            for i in range(len(F_CALIB)):
                _t = calib_result_raw[8 * (calibPt - 1) + i + 1].split('=')[1]
                if i in [4, 7]: _t = int(_t)    # Boolean (valid flag)
                else: _t = float(_t)            # Rest of fields
                calib_result[F_CALIB[i]].append(_t)
        calib_result = pd.DataFrame.from_dict(calib_result)
    
    # Logged gazepoints
    gp_data = []
    for _f in fn:
        with open(_f, 'r') as f:
            _ = f.readline()  # Skip first line (calib result)
            gp_log = {val:[] for val in F_GAZE}
            while True:
                eye = f.readline()
                if not eye: break   # EOF
                eye = eye.replace('"', '').split()[1:-1]
                for i, _t in enumerate(eye):
                    gp_log[F_GAZE[i]].append(F_GAZE_TYPE[i](_t.split("=")[1]))
            df = pd.DataFrame.from_dict(gp_log)
            # Convert: [0:1, 1:0] -> [-W/2:W/2, -H/2:H/2]
            df[['FPOGX', 'LPOGX', 'RPOGX', 'BPOGX']] -= 0.5
            df[['FPOGX', 'LPOGX', 'RPOGX', 'BPOGX']] *= MONITOR[X]
            df[['FPOGY', 'LPOGY', 'RPOGY', 'BPOGY']] *= -1
            df[['FPOGY', 'LPOGY', 'RPOGY', 'BPOGY']] += 0.5
            df[['FPOGY', 'LPOGY', 'RPOGY', 'BPOGY']] *= MONITOR[Y]
            gp_data.append(df)
    gp_data = pd.concat(gp_data, axis=0).reset_index(drop=True)
    return gp_data, calib_result


def load_raw_mouse_data(username):
    """Return merged pd.DataFrame mouse data"""
    fn = glob.glob(PATH_DATA_EXP_RAW_MOUSE % username)
    fn.sort()
    return pd.concat([pd.read_csv(f) for f in fn]).reset_index(drop=True)


def load_raw_synced_mouse_gaze_data(username):
    """Return mouse and gazepoint data with UTC time added to gazepoint"""
    d_mouse = load_raw_mouse_data(username)
    # There are some missing timestamps, remove them
    d_mouse = d_mouse[d_mouse.CPU_TICK >= d_mouse.CPU_TICK[0]]
    d_mouse["DY"] *= -1
    d_gaze, d_calib = load_raw_gazepoint_data(username)
    # Add UNIX timezone to gazepoint data
    t_utc = d_mouse["UTC_TIME"].to_numpy()
    t_cpu = d_mouse["CPU_TICK"].to_numpy()
    _a = (t_utc[-1] - t_utc[0]) / (t_cpu[-1] - t_cpu[0])
    _b = t_utc.mean() - _a * t_cpu.mean()
    d_gaze.insert(
        0, 
        "UTC_TIME", 
        (_a * d_gaze["TIME_TICK"].to_numpy() + _b) - GP3_LATENCY
    )
    return d_mouse, d_gaze, d_calib


if __name__ == "__main__":
    m, g, c = load_raw_synced_mouse_gaze_data('KKW')
    