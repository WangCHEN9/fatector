from loguru import logger
from fatector.machine_learning.Classification import Classification
from fatector.data_preprocess.FeatureEngineering import FeatureEngineering
from fatector.data_preprocess.DataLoader import DataLoader
from fatector.help_functions import get_config
import pandas as pd
from pathlib import Path


class Fatector:
    """main class to run preprocessing, training and inference"""

    def __init__(
        self, df_train: pd.DataFrame = None, df_test: pd.DataFrame = None
    ) -> None:
        """init function

        Args:
            df_train (pd.DataFrame, optional): df_train. Defaults to None.
            df_test (pd.DataFrame, optional): df_test. Defaults to None.
        """
        logger.add("./log/file_{time}.log", rotation="00:00")
        self.cfg = get_config()
        self.df_train = df_train
        self.df_test = df_test
        self.ml: Classification = None

    def _preprocess_data_from_csv(self, csv_file: Path, save_path: Path) -> None:
        """preprocess raw data from csv

        Args:
            csv_file (Path): csv file path
            save_path (Path): processed df save path
        """
        df = DataLoader().load_from_csv(csv_file)
        fe = FeatureEngineering(df)
        fe.add_features()
        fe.save_df_to_excel(save_path)

    def preprocess_train_test_data(self) -> None:
        """wrap to preprocess train and test raw data"""
        self._preprocess_data_from_csv(
            csv_file=self.cfg.data.raw_csv_train,
            save_path=self.cfg.data.processed_excel_train,
        )
        self._preprocess_data_from_csv(
            csv_file=self.cfg.data.raw_csv_test,
            save_path=self.cfg.data.processed_excel_test,
        )

    def load_processed_data_for_training(self):
        """load processed data from excel for training"""
        try:
            self.df_train = pd.read_excel(self.cfg.data.processed_excel_train)
            self.df_test = pd.read_excel(self.cfg.data.processed_excel_test)
        except Exception as err:
            logger.error(f"please run preprocess_train_test_data first !")
            logger.error(err)
        self.df_train.drop(
            columns=self.cfg.ml.column_to_drop_for_training, inplace=True
        )
        self.df_test.drop(columns=self.cfg.ml.column_to_drop_for_training, inplace=True)

    def training(self):
        """entry function for run automl for training"""
        if not self.df_train:
            self.load_processed_data_for_training()
        self.ml = Classification(df=self.df_train, df_test=self.df_test)
        self.ml.run_pycaret_experiment()
        self.ml.save_model()

    def inference(self):
        """run inference in test data, and save output"""
        self.model = Classification.load_model()
        self.df_test = pd.read_excel(self.cfg.data.processed_excel_test)
        df_test_for_inference = self.df_test.drop(
            columns=[self.cfg.data.target], inplace=False
        )
        prediction_of_probability = self.model.predict_proba(df_test_for_inference)
        self.df_test["prob_not_fake"] = prediction_of_probability[:, 0]
        self.df_test["prob_fake"] = prediction_of_probability[:, 1]
        self.df_test.to_excel(self.cfg.data.inference_result_excel, index=False)
        logger.info(
            f"run predict on data --> Done, and saved to {self.cfg.data.inference_result_excel}!"
        )


if __name__ == "__main__":
    fatector = Fatector()
    # fatector.preprocess_train_test_data()  #! only need run once
    fatector.training()  #! this will run automl, and save all runs to /mlruns
    fatector.inference()
