import yaml
import pickle
import time
import os
import torch
import numpy as np
# from box import Box
from datetime import datetime
import json
import hashlib
from collections.abc import Mapping, Sequence, Set

# HAS_BOX = True


# -----------------------------
# Config / YAML utilities
def load_yaml_config(config_path):
    """Load a YAML config file and wrap it with Box."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return config_yaml


def save_dict_to_yaml(data, filename, blank_line_between_top_keys=False):
    """Save a dict to a YAML file.

    Parameters
    ----------
    blank_line_between_top_keys : bool
        When True, insert one blank line between each top-level key block
        (useful for user_profile.yaml where each key is a separate profile).
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    yaml_data = yaml.dump(data, allow_unicode=True, sort_keys=False)

    if blank_line_between_top_keys and isinstance(data, dict) and len(data) > 1:
        # Top-level keys in YAML start at column 0 with no leading spaces.
        # Insert a blank line before each top-level key except the first.
        keys = list(data.keys())
        lines = yaml_data.splitlines(keepends=True)
        out = []
        first_key_seen = False
        for line in lines:
            # A top-level key line starts with a non-space, non-comment char
            # and matches one of the known top-level keys.
            stripped = line.rstrip()
            is_top_key = (
                stripped
                and not stripped.startswith(" ")
                and not stripped.startswith("#")
                and any(stripped.startswith(f"{k}:") for k in keys)
            )
            if is_top_key and first_key_seen:
                out.append("\n")
            if is_top_key:
                first_key_seen = True
            out.append(line)
        yaml_data = "".join(out)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(yaml_data)


# -----------------------------
# Pickle / NPZ utilities
def pickle_save(filename, data, try_multiple_save=100, verbose=False):
    """Save data to a pickle file, retrying on transient failures."""
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    if try_multiple_save <= 0:
        try_multiple_save = 1

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for _ in range(try_multiple_save):
        try:
            with open(filename, "wb") as fp:
                pickle.dump(data, fp)
            return
        except Exception as e:
            if verbose:
                print(f"Save attempt failed with error: {e}. Retrying...")
            time.sleep(0.5)
            continue
    raise ValueError("Save failed after multiple attempts. Check file directory and permissions.")


def pickle_load(file, verbose=False):
    """Load and return data from a pickle file."""
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    if verbose:
        print(f"Pickle file {file} loaded; datatype {str(type(file))}")
    return data


def npz_save(filename, **data_kwargs):
    """Save arrays to a compressed NPZ file."""
    if not filename.endswith(".npz"):
        filename += ".npz"
    np.savez_compressed(filename, **data_kwargs)


def npz_load(filename):
    """Load arrays from a NPZ file."""
    return np.load(filename, allow_pickle=True)


# -----------------------------
# Time / naming utilities
def format_large_number(num):
    """Format large integers with K/M/B suffixes (e.g., 2500000 -> 2.5M).

    Uses integer arithmetic to avoid float rounding issues — e.g.
    3_100_000 → "3.1M"  and  3_150_000 → "3.15M"  (not both "3.1M").
    Returns the original number as a string if it is < 1000.
    """
    if not isinstance(num, int) or num < 1000:
        return str(num)

    suffixes = ['', 'K', 'M', 'B', 'T']
    magnitude = 0
    divisor = 1

    while num >= divisor * 1000 and magnitude < len(suffixes) - 1:
        divisor   *= 1000
        magnitude += 1

    integer_part = num // divisor
    remainder    = num  % divisor

    if remainder == 0:
        return f"{integer_part}{suffixes[magnitude]}"

    # Exact fractional digits via integer arithmetic (no float precision loss)
    frac_digits = len(str(divisor)) - 1          # e.g. divisor=1_000_000 → 6
    frac_str = str(remainder).zfill(frac_digits).rstrip('0')
    return f"{integer_part}.{frac_str}{suffixes[magnitude]}"


def get_current_time_digit():
    """Return current time as MMDDHHMM string."""
    now = datetime.now()
    return now.strftime("%m%d%H%M")


def get_compact_timestamp_str(omit_year=False):
    """Return a compact timestamp string, optionally omitting the year."""
    now = datetime.now()
    year = now.year % 100
    day_of_year = now.timetuple().tm_yday
    total_seconds = now.hour * 3600 + now.minute * 60 + now.second
    if omit_year:
        session_name = f"{day_of_year:03d}{int_to_base26_char(total_seconds)}"
    else:
        session_name = f"{year:02d}{day_of_year:03d}{int_to_base26_char(total_seconds)}"
    return session_name


def int_to_base26_char(n):
    """Convert a non-negative integer into a 4-letter base-26 string."""
    base = 65 # ord("A")
    result = ""
    for i in range(3, -1, -1):
        value = (n // (26 ** i)) % 26
        result += chr(base + value)
    return result


# -----------------------------
# Configuration hashing
def _canonicalize(obj, float_rounding=6):
    """Canonicalize config objects into deterministic JSON-serializable forms."""
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]

    if isinstance(obj, Set) and not isinstance(obj, (str, bytes)):
        items = [_canonicalize(v) for v in obj]
        return sorted(items, key=lambda x: repr(x))

    try:
        import numpy as np
        if isinstance(obj, np.generic):
            obj = obj.item()
    except ImportError:
        pass

    if isinstance(obj, float):
        return round(obj, float_rounding)

    return obj


def config_hash(config, algo="sha256", length=16):
    """Return a stable hash string for a nested config structure."""
    canon = _canonicalize(config)
    serialized = json.dumps(
        canon,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    h = hashlib.new(algo)
    h.update(serialized.encode("utf-8"))
    return h.hexdigest()[:length]
    

if __name__ == "__main__":
    print(format_large_number(3150000))

