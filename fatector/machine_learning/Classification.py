import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret import classification as pyclf
from pathlib import Path
from typing import List
from loguru import logger
from fatector.help_functions import get_config


class Classification:
    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame = None,
    ):
        self.cfg = get_config()
        self.normalize = self.cfg.ml.normalize
        self.model_path = Path(self.cfg.ml.model_path)
        self.optimize = self.cfg.ml.pycaret_optimize
        self.model_name = self.cfg.ml.model_name
        self.y = self.cfg.data.target
        self.model_type = "classification"
        self.df = df
        self.df_test = df_test
        self.model_path.mkdir(exist_ok=True, parents=True)

    def run_pycaret_experiment(self):
        logger.info(f"running experiment for x = {self.df.columns}, y = {self.y}...")
        self._set_exp()
        top3 = pyclf.compare_models(turbo=False, n_select=3, sort=self.optimize)
        tuned_top3 = [
            pyclf.tune_model(i, n_iter=50, optimize=self.optimize) for i in top3
        ]
        calibrated_top3 = [pyclf.calibrate_model(i) for i in tuned_top3]
        logger.info(f"experiment finished !")
        self.best_model = pyclf.automl(optimize=self.optimize)
        self._get_config_from_exp()

    def _set_exp(self):
        experiment = pyclf.setup(
            self.df,
            target=self.y,
            test_data=self.df_test,
            fold_strategy="kfold",
            fold=10,
            fold_shuffle=True,
            train_size=0.7,
            normalize=self.normalize,
            remove_perfect_collinearity=False,
            silent=True,
            verbose=False,
            log_experiment=True,
            experiment_name=f"Clf-{self.optimize}-Norm-{self.normalize}-Y={self.y}-X=({len(self.df.columns)},{len(self.df)})",
            log_plots=["error", "confusion_matrix", "auc", "parameter"],
            log_profile=False,
            log_data=False,
            feature_ratio=False,
            # SMOTE (Synthetic Minority Over-sampling Technique) is applied by default to create synthetic datapoints for minority class.
            fix_imbalance=True,
        )

    def _get_config_from_exp(self):
        self.X = pyclf.get_config("X")
        self.y = pyclf.get_config("y")
        self.X_train = pyclf.get_config("X_train")
        self.y_train = pyclf.get_config("y_train")
        self.X_test = pyclf.get_config("X_test")
        self.y_test = pyclf.get_config("y_test")

    def save_model(self):
        save_file_path = self.model_path / f"{self.model_name}"
        pyclf.save_model(self.best_model, save_file_path)

    def load_pycaret_experiment(self):
        self._set_exp()
        self._get_config_from_exp()

    @classmethod
    def load_model(cls):
        cfg = get_config()
        save_file_path = Path(cfg.ml.model_path) / cfg.ml.model_name
        if save_file_path.is_file():
            return pyclf.load_model(save_file_path)
        else:
            logger.error(f"can't find model from {save_file_path}")
