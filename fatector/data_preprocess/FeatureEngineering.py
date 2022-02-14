from pathlib import Path
import pandas as pd
from loguru import logger
from fatector.help_functions import get_config
from fatector.data_preprocess.DataLoader import DataLoader
from pandas_profiling import ProfileReport


class FeatureEngineering:
    def __init__(self, df: pd.DataFrame) -> None:
        self.cfg = get_config()
        logger.info(self.cfg)
        self._df = df

    @property
    def df(self):
        return self._df

    def _add_count_for_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.cfg.data.features.count_of_columns:
            temp = df[column].agg("count")
            df[f"count_of_{column}"] = temp
        return df

    def _add_unique_for_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.cfg.data.features.nunique_of_columns:
            temp = df[column].agg("nunique")
            df[f"nunique_of_{column}"] = temp
        return df

    def _add_top_value_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.cfg.data.features.top_count_percentage:
            top_count = df[column].value_counts().max()
            df[f"top_count_{column}_ratio"] = top_count / len(df)
        return df

    def _add_event_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        count_dic = df[self.cfg.data.event].value_counts().to_dict()
        for event in self.cfg.data.features.percentage_of_event:
            df[f"{event}_percentage"] = count_dic.get(event, 0) / len(
                df
            )  # if event is not found, return 0
        return df

    def _add_new_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_count_for_columns(df)
        logger.debug(f"_add_count_for_columns -> Done")

        df = self._add_unique_for_columns(df)
        logger.debug(f"_add_unique_for_columns -> Done")

        df = self._add_top_value_percentage(df)
        logger.debug(f"_add_top_value_percentage -> Done")

        df = self._add_event_percentage(df)
        logger.debug(f"_add_event_percentage -> Done")
        df.drop_duplicates(
            subset=self.cfg.data.group_by,
            keep="first",
            inplace=True,
            ignore_index=True,
        )
        logger.info(
            f"remove duplicated rows"
        )  # ? a better way is to avoid this in the first place?
        return df

    def add_features(self) -> pd.DataFrame:
        df = self._df.groupby(self.cfg.data.group_by).apply(self._add_new_columns)
        self._df = df
        return df

    def save_df_to_excel(self) -> None:
        self._df.to_excel(self.cfg.data.training_excel_path, index=False)
        logger.info(f"saved df to excel {self.cfg.data.training_excel_path}")


if __name__ == "__main__":
    csv_file = Path(r"D:\03_Data_Scientist\fatector\data\fake_users.csv")
    df = DataLoader().load_from_csv(csv_file)

    fe = FeatureEngineering(df)
    df = fe.add_features()
    print(len(df))
    fe.save_df_to_excel()

    df = pd.read_excel(
        r"D:\03_Data_Scientist\fatector\data\training_data_with_added_features.xlsx"
    )

