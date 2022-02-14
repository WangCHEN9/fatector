import pytest
from fatector.data_preprocess.FeatureEngineering import FeatureEngineering
from fatector.help_functions import get_config
import pandas as pd

cfg = get_config()
df = pd.read_csv(cfg.data.raw_csv_test)

fe = FeatureEngineering(df)

def test_add_count_for_columns():
    df_1 = fe._add_count_for_columns(df)
    assert isinstance(df_1, pd.DataFrame)
    assert ("count_of_Event" in df.columns) == True
