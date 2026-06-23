from pathlib import Path

from ..utils.myutils import load_yaml_config

root_dir = Path(__file__).parent.parent.parent.resolve() / "configs"


def load_yaml_config_folder(folder_name: str):
	"""Load all YAML files in a target folder into a dict keyed by file stem."""
	target_dir = root_dir / folder_name

	configs = {}

	if not target_dir.exists():
		return configs

	for path in sorted(target_dir.glob("*.yml")) + sorted(target_dir.glob("*.yaml")):
		configs[path.stem] = load_yaml_config(str(path))

	return configs


def load_yaml_config_file(file_name: str):
	"""Load a single YAML config file under configs/."""
	target_path = root_dir / file_name

	if not target_path.exists() or not target_path.is_file():
		return {}

	return load_yaml_config(str(target_path))


def load_yaml_configs(folder_name: str):
	"""Backward-compatible alias for `load_yaml_config_folder`."""
	return load_yaml_config_folder(folder_name)


CFG_AGENT_PROFILE = load_yaml_config_folder("ans_agent")
CFG_AGENT_RL = load_yaml_config_folder("ans_rl")
CFG_AMORTIZED_INFERENCE = load_yaml_config_folder("ans_inference")

# Path to user_profile.yaml for runtime updates (session progress, etc.)


