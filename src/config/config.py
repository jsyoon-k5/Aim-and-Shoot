import os
from pathlib import Path

from ..utils.myutils import load_config

ENV = load_config(os.path.join(Path(__file__).parent.parent, "config/ans_env.yaml"))
SIM = load_config(os.path.join(Path(__file__).parent.parent, "config/simulation.yaml"))
AGN = load_config(os.path.join(Path(__file__).parent.parent, "config/agent.yaml"))
RLC = load_config(os.path.join(Path(__file__).parent.parent, "config/rl_config.yaml"))
INF = load_config(os.path.join(Path(__file__).parent.parent, "config/inference.yaml"))