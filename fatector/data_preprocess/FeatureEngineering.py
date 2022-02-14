import pandas as pd
from loguru import logging
from help_functions import get_config


class FeatureEngineering():
    def __init__(self, df:pd.DataFrame) -> None:
        get_config()
        self._df = df
    

    @property
    def df(self):
        return self._df
    

    # features we need : count of click envent, count of category, top category percentage, % click_ad % send_email

    def get_count_of_event(self) -> pd.DataFrame:
        pass


