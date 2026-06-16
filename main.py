import logging
from pathlib import Path

import pandas as pd

from src.exploration import DataExplorer
from src.models import ModelTrainer
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
    data_variants, target_names = preprocessor.prepare_data()
    logger.info(
        f"[TASK 2] Data preparation completed. Generated {list(data_variants.keys())} variants."
    )

    logger.info("[TASK 3 & 4] Starting classification and evaluation...")
    all_results = []

    for variant_name, (X_train, X_val, y_train, y_val) in data_variants.items():
        logger.info(f"Testing variant: {variant_name}")

        trainer = ModelTrainer()
        trainer.train_all(X_train, y_train)

        variant_results = trainer.evaluate_all(
            X_val, y_val, variant_name, output_path, target_names
        )
        all_results.extend(variant_results)

    results_df = pd.DataFrame(all_results)
    print("\n--- FINAL RESULTS COMPARISON ---")
    print(f"{results_df.sort_values(by='Accuracy', ascending=False)}\n")
    results_df.to_csv(output_path / "final_comparison.csv", index=False)
    logger.info(
        "[TASK 3 & 4] Classification and evaluation completed. "
        "Final results saved to output/final_comparison.csv."
    )

    logger.info("Data analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
