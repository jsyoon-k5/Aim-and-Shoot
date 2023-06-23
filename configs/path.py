import sys
sys.path.append("..")
from utilities.utils import path_to_cousin

# Material & Data
PATH_MAT = path_to_cousin("materials", "")
PATH_DATA = path_to_cousin("expdata", "")
# PATH_DATA_EXP_RAW_MOUSE = f"{PATH_DATA}experiment/%s/%s_mouse.csv"
# PATH_DATA_EXP_RAW_GAZE = f"{PATH_DATA}experiment/%s/%s_gaze.csv"
PATH_DATA_EXP_RAW_MOUSE = f"{PATH_DATA}experiment/_raw/%s-*_m.csv"
PATH_DATA_EXP_RAW_GAZE = f"{PATH_DATA}experiment/_raw/%s-*_e.txt"
PATH_DATA_EXP_RAW_CALIB = f"{PATH_DATA}experiment/%s/%s_calibration.csv"
PATH_DATA_EXP_CROP_MOUSE = f"{PATH_DATA}experiment/%s/%s_%s_mouse.csv"
PATH_DATA_EXP_CROP_GAZE = f"{PATH_DATA}experiment/%s/%s_%s_gaze.csv"
PATH_DATA_EXP_RAW_DB = f"{PATH_DATA}experiment/_raw/Experiment_%s_%s_896fa563.db"
PATH_DATA_EXP_DB_PAC = f"{PATH_DATA}experiment/%s/%s_%s_table_pac.csv"
PATH_DATA_EXP_DB_SES = f"{PATH_DATA}experiment/%s/%s_%s_table_ses.csv"
PATH_DATA_EXP_DB_TTJ = f"{PATH_DATA}experiment/%s/%s_%s_table_ttj.csv"
PATH_DATA_EXP_DB_TAR = f"{PATH_DATA}experiment/%s/%s_%s_table_tar.csv"
PATH_DATA_EXP_DB_TRI = f"{PATH_DATA}experiment/%s/%s_%s_table_tri.csv"
PATH_DATA_EXP_DB_INFO = f"{PATH_DATA}experiment/%s/%s_%s_table_info.csv"
PATH_DATA_EXP_GAZE_CORRECTION = f"{PATH_DATA}experiment/%s/%s_%s_gaze_correction.csv"
PATH_DATA_EXP_TRAJ_PKL = f"{PATH_DATA}experiment/%s/traj/"
PATH_DATA_SUMMARY_PLAYER = f"{PATH_DATA}summary/individual/%s-%s.csv"

PATH_DATA_SUMMARY_MERGED = f"{PATH_DATA}summary/experiment_result.csv"
PATH_DATA_SUMMARY_MERGED_MSEQ = f"{PATH_DATA}summary/main_sequence.csv"
PATH_DATA_SUMMARY_INDV = f"{PATH_DATA}summary/individual/%s-%s.csv"
PATH_DATA_SUMMARY_ANOVA = f"{PATH_DATA}summary/experiment_result_anova_format.csv"
PATH_DATA_SUMMARY_MSEQ = f"{PATH_DATA}summary/main_seq/%s.pkl"


# Reinforcement Learning
PATH_RL_ROOT = path_to_cousin("models", "")
PATH_RL_LOG = f"{PATH_RL_ROOT}%s/log/"
PATH_RL_CHECKPT = f"{PATH_RL_ROOT}%s/checkpoint/"
PATH_RL_BEST = f"{PATH_RL_ROOT}%s/best/"
PATH_RL_INFO = f"{PATH_RL_ROOT}%s/%s.csv"


# Inference
PATH_INFER_ROOT = path_to_cousin("inference", "")
PATH_INFER_RESULT = f"{PATH_INFER_ROOT}result/%s-%s"
PATH_INFER_RANDOM = f"{PATH_INFER_ROOT}random_sample/%s_%s.csv"


# Image and Video
PATH_VIS_ROOT = path_to_cousin("visualization", "")
PATH_VIS_VIDEO = f"{PATH_VIS_ROOT}video/%s/"
PATH_VIS_EXP_VERIF = f"{PATH_VIS_ROOT}experiment/verif_raw/%s/"
PATH_VIS_EXP_TRAJ = f"{PATH_VIS_ROOT}experiment/trajectory/%s/"
PATH_VIS_EXP_GAZE_PROF = f"{PATH_VIS_ROOT}experiment/gaze_profile/%s/"
PATH_VIS_EXP_MSEQ = f"{PATH_VIS_ROOT}experiment/main_sequence/"
PATH_VIS_EXP_HAND_PROF = f"{PATH_VIS_ROOT}experiment/hand_profile/%s/"
PATH_VIS_INFER_RESULT = f"{PATH_VIS_ROOT}inference"


# Temporal directory (for some testing)
PATH_TEMP = path_to_cousin("temp", "")