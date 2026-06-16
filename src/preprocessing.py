import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, target_col: str = "Status"):
        self.df = df.copy()
        self.target_col = target_col
        self.label_encoders = {}

        self.df = self.df.drop(columns=["ID"])

    def prepare_data(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        target_le = LabelEncoder()
        y = target_le.fit_transform(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_train, X_val = self._handle_missing_values(X_train, X_val)
        X_train, X_val = self._encode_categorical(X_train, X_val)

        data_variants = {"baseline": (X_train, X_val, y_train, y_val)}

        X_train_std, X_val_std = self._apply_standardization(X_train, X_val)
        data_variants["standardized"] = (X_train_std, X_val_std, y_train, y_val)

        X_train_pca, X_val_pca = self._apply_pca(X_train_std, X_val_std)
        data_variants["pca"] = (X_train_pca, X_val_pca, y_train, y_val)

        return data_variants

    def _handle_missing_values(self, X_train, X_val):
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

        imputer_num = SimpleImputer(strategy="median")
        X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
        X_val[num_cols] = imputer_num.transform(X_val[num_cols])

        imputer_cat = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
        X_val[cat_cols] = imputer_cat.transform(X_val[cat_cols])

        return X_train, X_val

    def _encode_categorical(self, X_train, X_val):
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

        for col in cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_val[col] = le.transform(X_val[col])
            self.label_encoders[col] = le

        return X_train, X_val

    def _apply_standardization(self, X_train, X_val):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled

    def _apply_pca(self, X_train, X_val):
        pca = PCA(n_components=0.90, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        logger.info(f"PCA reduced dimensions to: {X_train_pca.shape[1]}")
        return X_train_pca, X_val_pca
