from loguru import logger
from fatector.machine_learning.Classification import Classification
from fatector.machine_learning.Explainer import Explainer
from fatector.data_preprocess.FeatureEngineering import FeatureEngineering
from fatector.data_preprocess.DataLoader import DataLoader
from fatector.help_functions import get_config
import pandas as pd
from pathlib import Path


class Fatector:
    def __init__(self, df_train=None, df_test=None) -> None:
        logger.add("./log/file_{time}.log", rotation="00:00")
        self.cfg = get_config()
        self.df_train = df_train
        self.df_test = df_test
        self.ml = None

    def _preprocess_data_from_csv(self, csv_file: Path, save_path: Path) -> None:
        df = DataLoader().load_from_csv(csv_file)
        fe = FeatureEngineering(df)
        fe.add_features()
        fe.save_df_to_excel(save_path)

    def preprocess_train_test_data(self) -> None:
        self._preprocess_data_from_csv(
            csv_file=self.cfg.data.raw_csv_train,
            save_path=self.cfg.data.processed_excel_train,
        )
        self._preprocess_data_from_csv(
            csv_file=self.cfg.data.raw_csv_test,
            save_path=self.cfg.data.processed_excel_test,
        )

    def load_processed_data(self):
        self.df_train = pd.read_excel(self.cfg.data.processed_excel_train)
        self.df_test = pd.read_excel(self.cfg.data.processed_excel_test)
        self.df_train.drop(columns=self.cfg.ml.column_to_drop_for_ml, inplace=True)
        self.df_test.drop(columns=self.cfg.ml.column_to_drop_for_ml, inplace=True)

    def training(self):
        if self.df_train:
            self.ml = Classification(df=self.df_train, df_test=self.df_test)
            self.ml.run_pycaret_experiment()
            self.ml.save_model()
        else:
            logger.error(f"df train is not loaded yet !!!")

    def run_explainer(self):
        explainer = Explainer(ml=self.ml)
        df_result = explainer.predict_on_data(df_test)
        excel_path = test_excel.parent / f"{test_excel.stem}_with_prediction.xlsx"
        df_result.to_excel(excel_path, index=False)

        explainer.dataset_level_explainer()
        explainer.instance_level_explainer(df_train.sample(1))

    def inference(self):
        self.model = Classification.load_model()
        if not self.df_test:
            self.df_test = pd.read_excel(test_excel)
        df_test_without_target = self.df_test.loc[
            :, self.df_test.columns != self.cfg.data.target
        ]
        self.df_test["prediction"] = self.model.predict(df_test_without_target)
        logger.info(f"run predict on data -> Done !")
        self.df_test.to_excel(self.cfg.data.inference_result_excel, index=False)


if __name__ == "__main__":
    fatector = Fatector()
    fatector.preprocess_train_test_data()
    # fatector.training()
    fatector.run_explainer()
    fatector.inference()
