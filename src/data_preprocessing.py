from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
	"""Simple DataPreprocessor for non-medical features.

	Usage:
		pre = DataPreprocessor(numeric_features=[...], categorical_features=[...])
		pre.fit(df)
		X_proc = pre.transform(df)
	"""

	def __init__(self, numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None):
		self.numeric_features = numeric_features or []
		self.categorical_features = categorical_features or []
		self.transformer: ColumnTransformer = None

	def _build_transformer(self) -> ColumnTransformer:
		num_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		])

		cat_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
		])

		transformer = ColumnTransformer([
			("num", num_pipeline, self.numeric_features),
			("cat", cat_pipeline, self.categorical_features),
		], remainder="drop")

		return transformer

	def fit(self, X: pd.DataFrame, y=None):
		self.transformer = self._build_transformer()
		# sklearn expects 2D input for ColumnTransformer
		self.transformer.fit(X)
		return self

	def transform(self, X: pd.DataFrame):
		if self.transformer is None:
			raise RuntimeError("Call fit(...) before transform(...)")
		return self.transformer.transform(X)

	def fit_transform(self, X: pd.DataFrame, y=None):
		self.fit(X, y)
		return self.transformer.transform(X)

