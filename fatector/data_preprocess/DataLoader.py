from pathlib import Path
from loguru import logging
import pandas as pd

class DataLoader:
    """this class load the data
    """

    def __init__(self) -> None:
        pass

    def load_from_csv(self, csv_file_path:Path) -> None:
        self._df = pd.read_csv(csv_file_path, index = 0, header=0)
        logging.info(f'loaded df from {csv_file_path}')

    @property
    def df(self) -> pd.DataFrame:
        return self._df