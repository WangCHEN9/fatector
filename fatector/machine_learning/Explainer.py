import dalex as dx
import pandas as pd
from pandas.core.frame import DataFrame
from fatector.machine_learning.Classification import Classification
from pathlib import Path
from fatector.help_functions import get_config
from loguru import logger


class Explainer:
    def __init__(self, ml: Classification):
        self.cfg = get_config()
        self.explainer_save_path = Path(self.cfg.ml.explainer_save_path)
        self.explainer_save_path.mkdir(parents=True, exist_ok=True)
        self.name = self.cfg.data.target
        self.ml = ml
        self.exp = dx.Explainer(
            self.ml.best_model,
            self.ml.X,
            self.ml.y,
            label=self.name,
            model_type=self.ml.model_type,
        )
        self.model_performance = self.exp.model_performance()
        logger.info(f"model_performance : {self.model_performance}")

    def instance_level_explainer(self, df_x):
        """do instance level explainer with df_x
        :param df_x: one input x
        :type df_x: pd.DataFrame
        """
        self._break_down_interactions(df_x)
        self._break_down(df_x)
        self._shap(df_x)
        self._ceteris_paribus_profiles(df_x)

    def _ceteris_paribus_profiles(self, df_x):
        cp = self.exp.predict_profile(df_x, type="ceteris_paribus")
        logger.info(f"prepare ceteris_paribus for one instance")
        fig = cp.plot(show=False)
        fig.write_image(
            self.explainer_save_path
            / f"{self.name}_ceteris_paribus_idx-{df_x.index.values}.png"
        )

    def _break_down_interactions(self, df_x):
        bdi = self.exp.predict_parts(
            df_x, type="break_down_interactions", interaction_preference=10
        )
        logger.info(f"prepare break_down_interactions for one instance")
        fig = bdi.plot(show=False)
        fig.write_image(
            self.explainer_save_path
            / f"{self.name}_break_down_interactions_idx-{df_x.index.values}.png"
        )

    def _break_down(self, df_x):
        bd = self.exp.predict_parts(
            df_x,
            type="break_down",
        )
        logger.info(f"prepare break_down for one instance")
        fig = bd.plot(show=False)
        fig.write_image(
            self.explainer_save_path
            / f"{self.name}_break_down_idx-{df_x.index.values}.png"
        )

    def _shap(self, df_x):
        shap = self.exp.predict_parts(
            df_x,
            type="shap",
        )
        logger.info(f"prepare shap for one instance")
        fig = shap.plot(show=False)
        fig.write_image(
            self.explainer_save_path / f"{self.name}_shap_idx-{df_x.index.values}.png"
        )

    def dataset_level_explainer(self):
        #! will take loooooong time, if X has lots of rows
        variables = self.ml.X.columns.to_list()
        partial_dependence_profiles = self.exp.model_profile(
            variables=variables, type="partial", variable_type="numerical", N=None
        )
        logger.info(f"prepare model profile for {variables}")
        fig = partial_dependence_profiles.plot(
            geom="profiles",
            show=False,
        )
        image_path = (
            self.explainer_save_path / f"{self.name}_partial_dependence_profiles.png"
        )
        fig.write_image(image_path)
        logger.info(f"write image for {image_path} -> Done !")

    def predict_on_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["prediction"] = self.ml.best_model.predict(data)
        logger.info(f"run predict on data -> Done !")
        return data

    @property
    def model_params(self):
        return self.ml.best_model.get_params()
