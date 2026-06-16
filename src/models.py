import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_estimators = {}

    def train_all(self, X_train, y_train):
        nb_params = {"var_smoothing": [1e-9, 1e-8, 1e-7]}
        self._run_grid_search("NaiveBayes", GaussianNB(), nb_params, X_train, y_train)

        dt_params = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        }
        self._run_grid_search(
            "DecisionTree",
            DecisionTreeClassifier(random_state=42),
            dt_params,
            X_train,
            y_train,
        )

        rf_params = {
            "n_estimators": [50, 100],
            "max_depth": [10, 20],
            "min_samples_leaf": [1, 2],
        }
        self._run_grid_search(
            "RandomForest",
            RandomForestClassifier(random_state=42),
            rf_params,
            X_train,
            y_train,
        )

    def _run_grid_search(self, name, model, params, X_train, y_train):
        logger.info(f"Tuning hyperparameters for {name}...")
        grid = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        self.best_estimators[name] = grid.best_estimator_
        logger.info(f"Best params for {name}: {grid.best_params_}")

    def evaluate_all(self, X_val, y_val, variant_name, output_dir, class_names):
        results = []
        for name, model in self.best_estimators.items():
            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            results.append(
                {
                    "Variant": variant_name,
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }
            )

            self._plot_confusion_matrix(y_val, y_pred, name, variant_name, output_dir, class_names)

        return results

    def _plot_confusion_matrix(
        self, y_true, y_pred, model_name, variant_name, output_dir, class_names
    ):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title(f"Confusion Matrix: {model_name} ({variant_name})")
        plt.ylabel("Actual Status")
        plt.xlabel("Predicted Status")
        plt.tight_layout()
        filename = f"cm_{variant_name}_{model_name}.png".lower()
        plt.savefig(output_dir / filename)
        plt.close()
