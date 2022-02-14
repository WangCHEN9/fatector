from pathlib import Path
from loguru import logger
import pandas as pd


class DataLoader:
    """this class load the data from source, in case of there will be other data source instead of csv"""

    def load_from_csv(self, csv_file_path: Path) -> pd.DataFrame:
        """load df from csv

        Args:
            csv_file_path (Path): csv file path

        Returns:
            pd.DataFrame: df
        """
        df = pd.read_csv(csv_file_path, header=0)
        logger.info(f"loaded df from {csv_file_path}")
        return df
