# Aim-and-Shoot

Code and processed assets for **Modeling Visually-Guided Aim-and-Shoot
Behavior in First-Person Shooters**.

- DOI: https://doi.org/10.1016/j.ijhcs.2025.103503

The project contains a computationally rational FPS aim-and-shoot agent trained
with SAC, processed empirical player data, amortized inference models for
estimating model parameters from human behavioral dataset, and visualization
tools for replaying simulation and empirical behavior.

## Repository Layout

```text
configs/
  ans_agent/default.yaml        Agent, task, observation, and action settings.
  ans_rl/default.yaml           SAC training settings.
  ans_inference/default.yaml    Amortized inference dataset and trainer settings.

data/
  ans_agent_models/
    ijhcs_default/              Bundled SAC agent model and saved agent config.
  amortized_inference/
    models/ijhcs_default/       Bundled amortized inference checkpoint.

empirical/
  summary.csv                   Processed trial summary table.
  saccade_summary.csv           Processed saccade summary table.
  trajectories_20hz.h5          Processed trajectory store at 20 Hz.
  trajectories_60hz.h5          Processed trajectory store at 60 Hz.
  ijhcs_inference_results.csv   Inferred model parameters per player/condition.

src/
  agent/                        Aim-and-shoot task, player agent, SAC training, simulator.
  amortized_inference/          Dataset generation, trainer, and inferer.
  datamanager/                  Processed empirical data loaders.
  scripts/                      Convenience scripts and GUI tools.
  visualizer/                   Panda3D/OpenCV rendering utilities.
```

Run commands from the repository root with module syntax, for example
`python -m src.agent.training_sac`. This keeps relative imports and config
paths consistent.

## Environment

The recommended development environment is a conda environment named `ans_env`. A
clean setup can be created as follows:

```bash
conda create -n ans_env python=3.11
conda activate ans_env

# Install PyTorch for your CPU/CUDA platform. Example for CUDA systems:
conda install -c pytorch -c nvidia pytorch pytorch-cuda

# Core scientific/runtime packages.
conda install -c conda-forge numpy pandas scipy matplotlib seaborn h5py pyyaml tqdm psutil joblib opencv

# Packages that are easier to keep as pip dependencies.
pip install -r requirements.txt
```

If you use CPU-only PyTorch or a different CUDA version, replace the PyTorch
install command with the command recommended at https://pytorch.org/get-started/.

## Existing Models

This checkout includes the default model names used by the scripts.

### Aim-and-Shoot Agent

Bundled model directory:

```text
data/ans_agent_models/ijhcs_default/
  agent_config.yaml
  train_config.yaml
  checkpoints/rl_model_20000000_steps.zip
```

Load it in Python:

```python
from src.agent.simulator import AimandShootSimulator

simulator = AimandShootSimulator(model_name="ijhcs_default", ckpt="latest")
simulator.simulate(
    num_simul=10,
    num_cpu=10,
    deterministic=True,
    save_trajectory=True,
)
summary = simulator.get_summarized_results()
print(summary.head())
```

`ckpt` accepts:

- `"latest"`: largest `rl_model_*_steps.zip` checkpoint.
- An integer such as `20000000`: `checkpoints/rl_model_20000000_steps.zip`.

Set model parameters by passing a `player_state_preset` shared by all simulated
episodes, or by including parameter keys in each episode preset:

```python
from src.agent.simulator import AimandShootSimulator

params = {
    "param_motor_noise": 0.20,
    "param_position_noise": 0.08,
    "param_speed_noise": 0.04,
    "param_clock_noise": 0.04,
    "param_succ_reward": 40.0,
    "param_fail_penalty": 12.0,
    "param_reward_decay": 40.0,
    "param_penalty_decay": 45.0,
}

simulator = AimandShootSimulator("ijhcs_default", ckpt="latest")
simulator.simulate(num_simul=20, num_cpu=10, player_state_preset=params)
```

Set task conditions with `env_preset_list`. The empirical summary rows can be
parsed into task presets with `ijhcs_summary_to_env_presets`; the example below
combines those task presets with the `params` dictionary from the previous
example:

```python
from src.agent.preset_converter import ijhcs_summary_to_env_presets
from src.agent.simulator import AimandShootSimulator
from src.datamanager.load_emp_data import IJHCSExpDataLoader

loader = IJHCSExpDataLoader(load_traj=False)
summary = loader.load(
    return_traj=False,
    expertise_group="professional",
    player_index=0,
).head(10)

task_presets = ijhcs_summary_to_env_presets(summary)
env_preset_list = [{**task, **params} for task in task_presets]

simulator = AimandShootSimulator("ijhcs_default", ckpt="latest")
simulator.simulate(env_preset_list=env_preset_list, num_cpu=10)
```

Each task preset can include keys such as `target_pos_monitor_mm`,
`target_pos_world`, `target_orbit_axis_deg`, `target_motion_dir_deg`,
`target_speed_deg_s`, `target_radius_deg`, and `camera_azel_deg`.

### Amortized Inference Model

Bundled inference model directory:

```text
data/amortized_inference/models/ijhcs_default/pts/
  config.yaml
  iter_0100.pt
```

Load it:

```bash
python -m src.amortized_inference.inferer --run ijhcs_default --iter 100
```

The command above validates that the checkpoint can be loaded. See
"Run Inference on Empirical Data" below for actual parameter inference.

## Run RL Training

Entry point:

```text
src/agent/training_sac.py
```

Run a new SAC training job with the current default configs:

```bash
python -m src.agent.training_sac --model_name_prefix ans_sac --num_cpu 10 --train_config_preset default --agent_class AnSPlayerAgentDefault --agent_config_preset default
```

Outputs are written to:

```text
data/ans_agent_models/<model_name>/
  agent_config.yaml
  train_config.yaml
  checkpoints/
  best/
  tensorboard/
```

The training configuration comes from `configs/ans_rl/default.yaml`:

- SAC policy: custom `ModulatedSACPolicy`.
- Network: MLP `[512, 512]`.
- Batch size: `2048`.
- Learning rate: `5e-5`.
- Discount factor: `0.9`.
- Total steps: `20,000,000`.

TensorBoard:

```bash
tensorboard --logdir data/ans_agent_models
```

Note: the CLI defaults in `training_sac.py` still reference legacy names, so
use the explicit arguments above unless you have added those legacy config
presets locally.

## Run Simulation and Rendering

### Simulation Only

```python
from src.agent.simulator import AimandShootSimulator

simulator = AimandShootSimulator("ijhcs_default", ckpt="latest")
simulator.simulate(num_simul=20, num_cpu=10, deterministic=True)
df = simulator.get_summarized_results()
print(df[["completion_time_ms", "shoot_result", "shoot_error_mm"]].head())
```

### Export Simulation Video

```python
from src.agent.simulator import AimandShootSimulator

simulator = AimandShootSimulator("ijhcs_default", ckpt="latest")
simulator.simulate(num_simul=3, num_cpu=1, deterministic=True, save_trajectory=True)

path = simulator.render_records_to_video(
    render_config={
        "width": 1920,
        "height": 1080,
        "fps": 60,
        "freeze_ms": 0,
    }
)
print(path)
```

The default output root is `data/video/<model_name>/`.

### Interactive GUI Simulator

```bash
python -m src.scripts.gui_simulator --model-name ijhcs_default --ckpt latest
```

The GUI exposes the eight model parameters, task conditions, random sampling
ranges, episode count, and in-memory playback without saving MP4 files.

## Amortized Inference Pipeline

The inference pipeline has three stages:

1. Generate simulator-based training and validation datasets.
2. Train the amortized inference network.
3. Load the trained checkpoint and infer parameters from empirical or simulated data.

Main files:

```text
src/amortized_inference/dataset.py
src/amortized_inference/trainer.py
src/amortized_inference/inferer.py
```

### 1. Generate Datasets

Use Python so you can choose dataset sizes explicitly:

```python
from src.amortized_inference.dataset import TrainingDataset, ValidationDataset

train_ds = TrainingDataset(config_preset="default", load_existing=True)
train_ds.generate(
    n_param_sets=1024,
    num_cpu=10,
    repeat=1,
    save_trajectory=False,
)

valid_ds = ValidationDataset(config_preset="default", load_existing=True)
valid_ds.generate(
    num_cpu=10,
    save_trajectory=False,
)
```

Dataset files are saved under a hash of the inference config:

```text
data/amortized_inference/datasets/<config_hash>/
  train/*.h5
  valid/*.h5
```

Important defaults from `configs/ans_inference/default.yaml`:

- Training episodes per parameter vector: `64`.
- Validation users: `100`.
- Validation trials per user: `256` in the config file.
- Simulator model: `ijhcs_default`, checkpoint `latest`.
- Static features are normalized feature-wise to `[-1, 1]`.
- Trajectory features are ignored by the current source config unless
  `normalize.traj.ignore` is changed to `false`.

The bundled `ijhcs_default` inference checkpoint was trained with its saved
`pts/config.yaml`, where trajectory input is enabled (`normalize.traj.ignore:
false`) and validation trials per user is `600`.

### 2. Train Inference Model

```bash
python -m src.amortized_inference.trainer --preset default --name my_inference_run --device auto --max-iter 100
```

Outputs are written to:

```text
data/amortized_inference/models/<run_name>/
  pts/config.yaml
  pts/iter_XXXX.pt
  board/
  results/
```

If `--name` points to an existing run, the trainer resumes from the latest
checkpoint by default. Add `--no-resume` to ignore existing checkpoints.

### 3. Run Inference on Empirical Data

Example: infer parameters for one player and condition using the bundled
`ijhcs_default` inference model.

```python
from src.datamanager.load_emp_data import IJHCSExpDataLoader
from src.amortized_inference.inferer import AmortizedInferer

loader = IJHCSExpDataLoader(traj_hz=60, load_traj=True)
summary, trajectories = loader.load(
    return_traj=True,
    expertise_group="professional",
    player_index=0,
    sensitivity_mode="habitual",
    target_color_level="white",
)

inferer = AmortizedInferer(run_name="ijhcs_default", iteration=100)
params = inferer.infer(summary, traj_data=trajectories)
print(params)

# Optional verification by re-simulation.
sim_summary = inferer.simulate_with_inferred(params, n_episodes=64, num_cpu=10)
print(sim_summary.head())
```

The returned dictionary contains:

```text
param_motor_noise
param_position_noise
param_speed_noise
param_clock_noise
param_succ_reward
param_fail_penalty
param_reward_decay
param_penalty_decay
```

Existing IJHCS inference results are stored in:

```text
empirical/ijhcs_inference_results.csv
```

This file has one parameter row for each player x sensitivity mode x target
color condition.

## Load Empirical Dataset

Main loader:

```text
src/datamanager/load_emp_data.py
```

Basic summary loading:

```python
from src.datamanager.load_emp_data import (
    IJHCSExpDataLoader,
    load_ijhcs_summary,
    load_ijhcs_saccade_summary,
)

summary = load_ijhcs_summary()
saccades = load_ijhcs_saccade_summary()
print(summary.shape)
print(saccades.shape)
```

Filtered trial summary:

```python
from src.datamanager.load_emp_data import IJHCSExpDataLoader

loader = IJHCSExpDataLoader(traj_hz=60, load_traj=False)
df = loader.load(
    return_traj=False,
    expertise_group="professional",
    player_index=1,
)
print(df.head())
```

Summary plus trajectories:

```python
loader = IJHCSExpDataLoader(traj_hz=60, load_traj=True)
summary, trajectories = loader.load(
    return_traj=True,
    expertise_group="amateur",
    player_index=10,
    sensitivity_mode="default",
)

first_trial = summary.iloc[0]
first_trajectory = trajectories[0]
print(first_trial["trajectory_key"])
print(first_trajectory.head())
```

By default, `IJHCSExpDataLoader` applies analysis filters:

- Exclude truncated trials.
- Exclude TCT and shoot-error outliers.
- Require sufficient gaze data.
- Require close initial gaze.
- Require non-missing gaze and hand reaction times.

Disable those filters with:

```python
loader = IJHCSExpDataLoader(traj_hz=60, load_traj=True, filtered=False)
summary, trajectories = loader.load(return_traj=True)
```

The local processed dataset currently contains `38,397` summary rows, with
`35,678` rows after the default quality filters. The trajectory files are large
and may be omitted from lightweight Git distributions; request the full data
from the author if they are missing.

## Empirical Replay and Comparison Videos

### Empirical Replay

Render empirical human trajectories directly from the 60 Hz trajectory store:

```bash
python -m src.visualizer.empirical_replay --filter expertise_group=professional --filter player_index=0 --limit 3
```

Default output root:

```text
empirical/video/
```

To avoid accidental huge renders, the function/script guards against rendering
more than `100` trials unless explicitly confirmed.

### Simulation vs Empirical Comparison

Run simulations under the same task conditions as empirical trials, inject the
corresponding inferred parameters from `empirical/ijhcs_inference_results.csv`,
select the best matching trials, and render side-by-side videos:

```bash
python -m src.scripts.compare_sim_empirical_video --model-name ijhcs_default --ckpt latest --top-n 10 --num-cpu 10
```

Useful filters:

```bash
python -m src.scripts.compare_sim_empirical_video --expertise-group professional --player-index 1 --sensitivity-mode habitual --target-color-level white --top-n 10
```

Outputs:

```text
empirical/video/sim_empirical_compare/
empirical/video/cache/sim_empirical_compare/
```

Cached simulation records are reused unless `--refresh-cache` is passed.

## Model Specification

### Inferred Model Parameters

The eight inferred/user-specific parameters are defined in
`configs/ans_inference/default.yaml` and the agent config:

| Parameter | Meaning | Range |
| --- | --- | --- |
| `param_motor_noise` | Motor execution noise | `0.0` to `0.5` |
| `param_position_noise` | Target position perception noise | `0.0` to `0.5` |
| `param_speed_noise` | Target speed perception noise | `0.0` to `0.5` |
| `param_clock_noise` | Internal timing noise | `0.0` to `0.5` |
| `param_succ_reward` | Success reward magnitude | `1.0` to `64.0` |
| `param_fail_penalty` | Failure penalty magnitude | `1.0` to `64.0` |
| `param_reward_decay` | Success reward temporal decay | `5.0` to `95.0` |
| `param_penalty_decay` | Failure penalty temporal decay | `5.0` to `95.0` |

### Task Conditions

Current source config (`configs/ans_agent/default.yaml`):

| Setting | Value |
| --- | --- |
| Monitor size | `531.3 x 298.8 mm`, `1920 x 1080 px` |
| Camera FOV | `103.0 deg` width, `70.533 deg` height |
| Mouse gain | `1.0 deg/mm` |
| Camera max deviation | `1.22 deg` |
| Target radius | `4.0` to `13.0 mm` |
| Target angular speed | `0.0` to `40.0 deg/s` |
| Target motion direction | `0.0` to `90.0 deg` |
| Target initial azimuth | `2.0` to `37.0 deg` |
| Target initial elevation | `1.0` to `21.0 deg` |
| Time limit | `3000 ms` |

The bundled `ijhcs_default` SAC model loads its saved
`data/ans_agent_models/ijhcs_default/agent_config.yaml`, not the source default
config. That saved config uses the same main task geometry but has
`target_radius_mm_range: 1.2` to `13.0` and `max_hand_speed_mm_s: 4000.0`.

### Player and Action Settings

Default player priors:

- Head position: normal distribution, mean `[0.0, 77.9, 575.0]` mm, bounded by
  `[-65.0, -42.1, 408.0]` and `[65.0, 197.9, 742.0]`.
- Gaze position: radial normal with standard deviation `6.19569`.
- Hand reaction time: skew-normal, `100` to `300 ms`, mean `131.6 ms`.
- Gaze reaction time: skew-normal, `50` to `300 ms`, mean `80.1 ms`.
- Max camera speed: `300 deg/s`.

Action dimensions:

| Action | Range | Description |
| --- | --- | --- |
| `th` | `100` to `2000` | Planned hand movement duration/timing parameter. |
| `kc` | `0.0` to `1.0` | Shoot/click control, threshold `0.5`. |
| `tc` | `0.0` to `1.0` | Hand trajectory control. |
| `kg` | `0.0` to `1.0` | Gaze control. |

Observation features include target future position, target velocity by orbit,
target radius, hand velocity, gaze position, head position, and the eight
policy-conditioning model parameters.

## Outputs and Cache Directories

Common generated outputs:

```text
data/ans_agent_models/<run_name>/          SAC model checkpoints/logs.
data/amortized_inference/datasets/         Generated inference datasets.
data/amortized_inference/models/<run>/     Inference checkpoints/logs/results.
data/video/                                Simulation videos.
empirical/video/                           Empirical replay/comparison videos.
```

Panda3D rendering uses a user-local or temporary cache directory configured in
`src/visualizer/renderer.py`, so normal rendering should not create
`Panda3D-*` cache folders in the cloned repository.

## Troubleshooting

- Use `python -m ...` from the repository root. Directly running a file can
  break relative imports.
- If empirical trajectory loading returns empty DataFrames, check that
  `empirical/trajectories_20hz.h5` and/or `empirical/trajectories_60hz.h5`
  exist.
- If `AimandShootSimulator("ijhcs_default")` fails, check that
  `data/ans_agent_models/ijhcs_default/agent_config.yaml` and at least one
  checkpoint zip exist.
- If inference loading fails, check that
  `data/amortized_inference/models/ijhcs_default/pts/iter_0100.pt` exists.
- Rendering requires Panda3D and OpenCV. Install `panda3d` with pip and
  `opencv`/`opencv-python` through your chosen package manager.

## Citation

Please cite this paper if you use this code or dataset in your research.

```bibtex
@article{yoon2025modeling,
  title={Modeling visually-guided aim-and-shoot behavior in first-person shooters},
  author={Yoon, June-Seop and Moon, Hee-Seung and Boudaoud, Ben and Spjut, Josef and Frosio, Iuri and Lee, Byungjoo and Kim, Joohwan},
  journal={International Journal of Human-Computer Studies},
  volume={199},
  pages={103503},
  year={2025},
  publisher={Elsevier}
}
```

For access to the large full raw data, contact `jsyoon.k5 at gmail.com`.
