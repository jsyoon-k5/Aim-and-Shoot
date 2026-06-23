import numpy as np
import torch

def sort_and_pad_traj_data(stat_data, traj_data, value=0):
    """Sort trajectories by length and pad them to the max length."""
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)

    padded_data = []
    for traj in traj_data:
        padded_data.append(
            np.pad(
                traj,
                ((0, max_len), (0, 0)),
                "constant",
                constant_values=value,
            )[:max_len]
        )

    lens = torch.LongTensor(traj_lens)
    lens, sorted_idx = lens.sort(descending=True)
    padded = max_len - lens

    sorted_trajs = torch.FloatTensor(np.array(padded_data))[sorted_idx]
    sorted_stats = torch.FloatTensor(np.array(stat_data))[sorted_idx]
    return sorted_stats, sorted_trajs, padded, sorted_idx


def mask_and_pad_traj_data(traj_data, value=0):
    """Pad trajectories and return a boolean mask for valid timesteps."""
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)
    mask = torch.zeros((len(traj_data), max_len))

    padded_data = []
    for i, traj in enumerate(traj_data):
        padded_data.append(
            np.pad(
                traj,
                ((0, max_len), (0, 0)),
                "constant",
                constant_values=value,
            )[:max_len]
        )
        mask[i, :traj_lens[i]] = 1

    padded_trajs = torch.FloatTensor(np.array(padded_data))
    return padded_trajs, mask.bool()


# -----------------------------
# Encoding utilities
def fourier_encode(x, max_freq, num_bands=4):
    """Fourier feature position encoding (Perceiver-style)."""
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[((None,) * (len(x.shape) - 1) + (...,))]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# -----------------------------
# Device utilities
def get_auto_device():
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)