import logging
from pathlib import Path

import pandas as pd

from src.exploration import DataExplorer
from src.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "cirrhosis.csv"
    output_path = base_path / "output"
    output_path.mkdir(exist_ok=True)

    logger.info("Starting data analysis pipeline...")

    # Loading data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info("Data loaded successfully.")

    # Task 1
    logger.info("[TASK 1] Exploring data...")
    explorer = DataExplorer(df, output_path)
    explorer.run_full_analysis()
    logger.info("[TASK 1] Data exploration completed. Results saved in the output directory.")

    # Task 2
    logger.info("[TASK 2] Preprocessing data...")
    preprocessor = DataPreprocessor(df)
    data_variants = preprocessor.prepare_data()

    logger.info(f"Data preparation completed. Generated {list(data_variants.keys())} variants.")

    x_tr, _, _, _ = data_variants["standardized"]
    logger.info(f"Standardized train set shape: {x_tr.shape}")
    logger.info("[TASK 2] Preprocessing data completed.")

    logger.info("Data analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
