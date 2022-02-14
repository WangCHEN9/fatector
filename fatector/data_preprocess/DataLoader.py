from pathlib import Path
from loguru import logger
import pandas as pd


class DataLoader:
    """this class load the data"""

    def __init__(self) -> None:
        pass

    def load_from_csv(self, csv_file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_file_path, header=0)
        logger.info(f"loaded df from {csv_file_path}")
        return df
