import hydra
import omegaconf


def get_config() -> dict:
    """return cfg dict according to hydra config setting

    Returns:
        dict: config dict
    """
    cfg = omegaconf.OmegaConf.load(r"./config.yaml")
    return cfg
