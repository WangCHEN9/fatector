import hydra
import omegaconf


def get_config():
    cfg = omegaconf.OmegaConf.load(r"./config.yaml")
    return cfg
