import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./", config_name="config")
def get_config(cfg: DictConfig) -> None:
    return OmegaConf.to_yaml(cfg)
