from pathlib import Path
import pandas as pd
from loguru import logger
from fatector.help_functions import get_config
from fatector.data_preprocess.DataLoader import DataLoader
from pandas_profiling import ProfileReport


class FeatureEngineering:
    """this class doing feature engineering, final df can be accessed by .df property"""

    def __init__(self, df: pd.DataFrame) -> None:
        """init function

        Args:
            df (pd.DataFrame): source dataframe
        """
        self.cfg = get_config()
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """return self._df

        Returns:
            pd.DataFrame: return current status of processed df
        """
        return self._df

    def _add_count_for_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """add count for requested columns

        Args:
            df (pd.DataFrame): df to process

        Returns:
            pd.DataFrame: processed df
        """
        for column in self.cfg.data.features.count_of_columns:
            temp = df[column].agg("count")
            df[f"count_of_{column}"] = temp
        return df

    def _add_unique_for_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """add unique count for requested columns

        Args:
            df (pd.DataFrame): df to process

        Returns:
            pd.DataFrame: processed df
        """
        for column in self.cfg.data.features.nunique_of_columns:
            temp = df[column].agg("nunique")
            df[f"nunique_of_{column}"] = temp
        return df

    def _add_top_value_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """add most often value's percentage

        Args:
            df (pd.DataFrame): df to process

        Returns:
            pd.DataFrame: processed df
        """
        for column in self.cfg.data.features.top_count_percentage:
            top_count = df[column].value_counts().max()
            df[f"top_count_{column}_ratio"] = top_count / len(df)
        return df

    def _add_event_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """add requested event's percentage

        Args:
            df (pd.DataFrame): df to process

        Returns:
            pd.DataFrame: processed df
        """
        count_dic = df[self.cfg.data.event].value_counts().to_dict()
        for event in self.cfg.data.features.percentage_of_event:
            df[f"{event}_percentage"] = count_dic.get(event, 0) / len(
                df
            )  # * if event not founded from count_dic, return 0
        return df

    def _add_new_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """call other feature engineering functions

        Args:
            df (pd.DataFrame): df to process

        Returns:
            pd.DataFrame: prcessed df
        """
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

    def add_features(self) -> None:
        """wrap function runs groupby and apply to implement feature engineering"""
        df = self._df.groupby(self.cfg.data.group_by).apply(self._add_new_columns)
        self._df = df

    def save_df_to_excel(self, excel_path: Path) -> None:
        """save df to defined path

        Args:
            excel_path (Path): save excel path
        """
        self._df.to_excel(excel_path, index=False)
        logger.info(f"saved df to excel {excel_path}")
