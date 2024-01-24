import sys
sys.path.append("..")
from utils.utils import path_to_cousin

# Material & Data
PATH_MAT = path_to_cousin("materials", "")
PATH_DATA = path_to_cousin("experiment", "")
# PATH_DATA_EXP_RAW_MOUSE = f"{PATH_DATA}raw/%s/%s_mouse.csv"
# PATH_DATA_EXP_RAW_GAZE = f"{PATH_DATA}raw/%s/%s_gaze.csv"
PATH_DATA_EXP_RAW_MOUSE = f"{PATH_DATA}raw/_raw/%s-*_m.csv"
PATH_DATA_EXP_RAW_GAZE = f"{PATH_DATA}raw/_raw/%s-*_e.txt"
PATH_DATA_EXP_RAW_CALIB = f"{PATH_DATA}raw/%s/%s_calibration.csv"
PATH_DATA_EXP_CROP_MOUSE = f"{PATH_DATA}raw/%s/%s_%s_mouse.csv"
PATH_DATA_EXP_CROP_GAZE = f"{PATH_DATA}raw/%s/%s_%s_gaze.csv"
PATH_DATA_EXP_RAW_DB = f"{PATH_DATA}raw/_raw/Experiment_%s_%s_896fa563.db"
PATH_DATA_EXP_DB_PAC = f"{PATH_DATA}raw/%s/%s_%s_table_pac.csv"
PATH_DATA_EXP_DB_SES = f"{PATH_DATA}raw/%s/%s_%s_table_ses.csv"
PATH_DATA_EXP_DB_TTJ = f"{PATH_DATA}raw/%s/%s_%s_table_ttj.csv"
PATH_DATA_EXP_DB_TAR = f"{PATH_DATA}raw/%s/%s_%s_table_tar.csv"
PATH_DATA_EXP_DB_TRI = f"{PATH_DATA}raw/%s/%s_%s_table_tri.csv"
PATH_DATA_EXP_DB_INFO = f"{PATH_DATA}raw/%s/%s_%s_table_info.csv"
PATH_DATA_EXP_GAZE_CORRECTION = f"{PATH_DATA}raw/%s/%s_%s_gaze_correction.csv"
PATH_DATA_EXP_TRAJ_PKL = f"{PATH_DATA}raw/%s/traj/"
PATH_DATA_SUMMARY_PLAYER = f"{PATH_DATA}summary/individual/%s-%s.csv"

PATH_DATA_SUMMARY = f"{PATH_DATA}summary/"
PATH_DATA_SUMMARY_MERGED = f"{PATH_DATA}summary/experiment_result.csv"
PATH_DATA_SUMMARY_MERGED_MSEQ = f"{PATH_DATA}summary/main_sequence.csv"
PATH_DATA_SUMMARY_ANOVA = f"{PATH_DATA}summary/experiment_result_anova_format.csv"


# Reinforcement Learning
PATH_RL_ROOT = path_to_cousin("models", "")
PATH_RL_LOG = f"{PATH_RL_ROOT}%s/log/"
PATH_RL_CHECKPT = f"{PATH_RL_ROOT}%s/checkpoint/"
PATH_RL_BEST = f"{PATH_RL_ROOT}%s/best/"
PATH_RL_INFO = f"{PATH_RL_ROOT}%s/%s"


# Inference
PATH_INFER_ROOT = path_to_cousin("inference_iter", "")
PATH_INFER_RESULT = f"{PATH_INFER_ROOT}result-%s/%s-%s"
PATH_INFER_RANDOM = f"{PATH_INFER_ROOT}random_sample/%s_%s.csv"

PATH_AMORT_ROOT = path_to_cousin("amortizer", "")
PATH_AMORT_SIM_DATASET = f"{PATH_AMORT_ROOT}data/datasets/"
PATH_AMORT_MODEL = f"{PATH_AMORT_ROOT}data/amortizer_models"
PATH_AMORT_BOARD = f"{PATH_AMORT_ROOT}data/board"
PATH_AMORT_RESULT = f"{PATH_AMORT_ROOT}data/results"

PATH_AMORT_EVAL_ALL = f"{PATH_AMORT_ROOT}evaluation/"
PATH_AMORT_EVAL = f"{PATH_AMORT_ROOT}evaluation/%s/%s/%03d/"


# Image and Video
PATH_VIS_ROOT = path_to_cousin("visualization", "")
PATH_VIS_VIDEO = f"{PATH_VIS_ROOT}video/%s/"
PATH_VIS_EXP_VERIF = f"{PATH_VIS_ROOT}experiment/verif_raw/%s/"
PATH_VIS_EXP_TRAJ = f"{PATH_VIS_ROOT}experiment/trajectory/%s/"
PATH_VIS_EXP_GAZE_PROF = f"{PATH_VIS_ROOT}experiment/gaze_profile/%s/"
PATH_VIS_EXP_MSEQ = f"{PATH_VIS_ROOT}experiment/main_sequence/"
PATH_VIS_EXP_HAND_PROF = f"{PATH_VIS_ROOT}experiment/hand_profile/%s/"
PATH_VIS_EXP_CAM_PROF = f"{PATH_VIS_ROOT}experiment/camera_profile/%s/"
PATH_VIS_INFER_RESULT = f"{PATH_VIS_ROOT}inference"
PATH_VIS_MODULATION = f"{PATH_VIS_ROOT}modulation"
PATH_VIS_ANOVA_BAR = f"{PATH_VIS_ROOT}experiment/anova/"
PATH_VIS_EXP_DISTANCE = f"{PATH_VIS_ROOT}experiment/distance/"
PATH_VIS_FITTS_EXP = f"{PATH_VIS_ROOT}experiment/fitts_law/"
PATH_VIS_AMORT_RESULT = f"{PATH_VIS_ROOT}amortization/"
PATH_VIS_AMORT_COMPARE = f"{PATH_VIS_ROOT}amort_compare/"
PATH_VIS_EXPLORE = f"{PATH_VIS_ROOT}exploration/"


# Temporal directory (for some testing)
PATH_TEMP = path_to_cousin("temp", "")