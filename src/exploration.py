import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class DataExplorer:
    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = output_dir

    def run_full_analysis(self) -> None:
        self._save_basic_info()
        self._plot_target_distribution()
        self._plot_correlation_matrix()
        self._plot_missing_values()

    def _save_basic_info(self) -> None:
        stats = self.df.describe(include="all")
        stats.to_csv(self.output_dir / "basic_statistics.csv")

    def _plot_target_distribution(self) -> None:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x="Status", palette="viridis", hue="Status", legend=False)
        plt.title("Target variable distribution (Status)")
        plt.xlabel("Status")
        plt.ylabel("Count")
        plt.savefig(self.output_dir / "target_distribution.png")
        plt.close()

    def _plot_correlation_matrix(self) -> None:
        plt.figure(figsize=(12, 10))
        numeric_df = self.df.select_dtypes(include=["float64", "int64"])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="RdBu", fmt=".2f")
        plt.title("Numeric values correlation matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png")
        plt.close()

    def _plot_missing_values(self) -> None:
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isna(), yticklabels=False, cbar=False, cmap="viridis")
        plt.title("Missing values map")
        plt.savefig(self.output_dir / "missing_values_map.png")
        plt.close()
