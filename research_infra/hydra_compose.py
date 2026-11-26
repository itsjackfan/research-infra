from pathlib import Path

from hydra import compose as hydra_compose, initialize_config_dir


def compose(config_dir: Path, config_name="config", overrides=()):
    with initialize_config_dir(str(config_dir), version_base=None):
        return hydra_compose(config_name, overrides=list(overrides))
