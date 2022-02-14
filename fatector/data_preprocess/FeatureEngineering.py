from pathlib import Path
import pandas as pd
from loguru import logger
from fatector.help_functions import get_config
from fatector.data_preprocess.DataLoader import DataLoader


class FeatureEngineering:
    def __init__(self, df: pd.DataFrame) -> None:
        self.cfg = get_config()
        logger.info(self.cfg)
        self._df = df

    @property
    def df(self):
        return self._df

    def _add_count_for_columns(self) -> None:
        for column in self.cfg.data.features.count_of_columns:
            temp = self._df[column].agg("count")
            self._df[f"count_of_{column}"] = temp
            logger.info(f"added count for {column}")

    def _add_unique_for_columns(self) -> None:
        for column in self.cfg.data.features.nunique_of_columns:
            temp = self._df[column].agg("nunique")
            self._df[f"nunique_of_{column}"] = temp
            logger.info(f"added unique for {column}")

    def _add_top_value_percentage(self) -> None:
        for column in self.cfg.data.features.top_count_percentage:
            top_count = self._df[column].value_counts().max()
            self._df[f"top_count_{column}_ratio"] = top_count / len(self._df)
            logger.info(f"added top value percentage for {column}")

    def _add_event_percentage(self) -> None:
        count_dic = self._df[self.cfg.data.event].value_counts().to_dict()
        for event in self.cfg.data.features.percentage_of_event:
            self._df[f"{event}_percentage"] = count_dic.get(event) / len(self._df)
            logger.info(f"added event count for {event}")

    def _add_new_columns(self) -> None:
        self._add_count_for_columns()
        self._add_unique_for_columns()
        self._add_top_value_percentage()
        self._add_event_percentage()

    def add_features(self) -> None:
        df = self._df.groupby(self.cfg.data.group_by).apply(self._add_new_columns())
        print(df)


if __name__ == "__main__":
    csv_file = Path(r"D:\03_Data_Scientist\fatector\data\fake_users.csv")
    df = DataLoader().load_from_csv(csv_file)

    fe = FeatureEngineering(df)
    fe.add_features()
