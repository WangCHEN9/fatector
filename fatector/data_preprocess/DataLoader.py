from pathlib import Path
from loguru import logger
import pandas as pd


class DataLoader:
    """this class load the data"""

    def __init__(self) -> None:
        pass

    def load_from_csv(self, csv_file_path: Path) -> None:
        self._df = pd.read_csv(csv_file_path, index_col=0, header=0)
        logger.info(f"loaded df from {csv_file_path}")
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self._df
