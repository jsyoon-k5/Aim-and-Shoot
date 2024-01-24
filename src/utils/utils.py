import os, pickle, time
from datetime import datetime

import numpy as np
import torch

import sys

from torch._C import Value
sys.path.append('..')
from configs.experiment import *

def path_to_cousin(cousinname, filename):
    return '/'.join(
        (
            *os.path.abspath(__file__).split("\\")[:-2],
            cousinname,
            filename
        )
    )


def now_to_string(omit_year=False, omit_ms=False):
    ct = datetime.now()
    if omit_year:
        if omit_ms:
            return f"{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}"
        else:
            return f"{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}_{ct.microsecond//10000:02d}"
    if omit_ms:
        return f"{ct.year%100:02d}{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}"
    return f"{ct.year%100:02d}{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}_{ct.microsecond//10000:02d}"


def list2str(s, sep=','):
    return sep.join(list(map(str, s)))


def pickle_save(file, data, try_multiple_save=100):
    if not file.endswith('.pkl'):
        file += '.pkl'
    if try_multiple_save <= 0: try_multiple_save = 1
    for _ in range(try_multiple_save):
        try:
            with open(file, "wb") as fp:
                pickle.dump(data, fp)
            return
        except:
            time.sleep(0.5)
            continue
    raise ValueError("Save failed. Check file directory.")


def pickle_load(file):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    return data


def anonymization(player):
    if player in PLAYER["AMA"]:
        return f"A{PLAYER['AMA'].index(player)+1}"
    elif player in PLAYER["PRO"]:
        return f"P{PLAYER['PRO'].index(player)+1}"
    else:
        raise ValueError("Invalid player name.")


def save_string_to_file(path, string_data):
    with open(path, 'w') as file:
        file.write(string_data)
    


########################################################
### https://github.com/hsmoon121/amortized-inference-hci
def sort_and_pad_traj_data(stat_data, traj_data, value=0):
    """
    Sort and pad trajectory data based on their lengths.

    stat_data (ndarray): static data with a shape (num_data, stat_feature_dim).
    traj_data (list): List of trajectory data, each item should have a shape (traj_length, traj_feature_dim).
    value (int, optional): Padding value, default is 0.
    ---
    outputs (tuple): Tuple containing sorted static data (torch.Tensor), sorted padded trajectory data (torch.Tensor),
               padding lengths (torch.Tensor), and sorted indices (torch.Tensor).
    """
    # Get trajectory lengths
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)

    # Pad trajectory data
    padded_data = []
    for traj in traj_data:
        padded_data.append(np.pad(
            traj,
            ((0, max_len), (0, 0)),
            "constant",
            constant_values=value
        )[:max_len])

    # Sort trajectory data based on length
    lens = torch.LongTensor(traj_lens)
    lens, sorted_idx = lens.sort(descending=True)
    padded = max_len - lens

    # Get sorted static and trajectory data
    sorted_trajs = torch.FloatTensor(np.array(padded_data))[sorted_idx]
    sorted_stats = torch.FloatTensor(np.array(stat_data))[sorted_idx]
    return sorted_stats, sorted_trajs, padded, sorted_idx


def mask_and_pad_traj_data(traj_data, value=0):
    """
    Create a mask and pad trajectory data based on their lengths.

    traj_data (list): List of trajectory data, each item should have a shape (traj_length, traj_feature_dim).
    value (int, optional): Padding value, default is 0.
    ---
    outputs (tuple): Tuple containing padded trajectory data (torch.Tensor) and mask (torch.BoolTensor).
    """
    # Get trajectory lengths
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)
    mask = torch.zeros((len(traj_data), max_len))

    # Pad trajectory data and create mask
    padded_data = []
    for i, traj in enumerate(traj_data):
        padded_data.append(np.pad(
            traj,
            ((0, max_len), (0, 0)),
            "constant",
            constant_values=value
        )[:max_len])
        mask[i, :traj_lens[i]] = 1

    # Get padded data and mask to tensors
    padded_trajs = torch.FloatTensor(np.array(padded_data))
    return padded_trajs, mask.bool()


def fourier_encode(x, max_freq, num_bands = 4):
    """
    Fourier feature postiion encodings
    reference: https://github.com/lucidrains/perceiver-pytorch
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[((None,) * (len(x.shape) - 1) + (...,))]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


def get_auto_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


if __name__ == "__main__":
    print(anonymization('what2d0'))