from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer


class ModelTrainer:
	def __init__(self, random_state: int = 42):
		self.random_state = random_state
		self.model: Optional[Pipeline] = None

	def build_pipeline(self, preprocessor: DataPreprocessor, feature_engineer: FeatureEngineer) -> Pipeline:
		# Wrapper pipeline: engineer -> preprocess -> model
		clf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)

		# We assume preprocessing will be applied to the engineered DataFrame externally for flexibility.
		self.model = Pipeline([("clf", clf)])
		return self.model

	def train(self, X: pd.DataFrame, y: Iterable[Any]):
		if self.model is None:
			raise RuntimeError("Call build_pipeline(...) before train(...) ")
		self.model.fit(X, y)
		return self.model

	def predict_proba(self, X: pd.DataFrame):
		return self.model.predict_proba(X)

	def predict(self, X: pd.DataFrame):
		return self.model.predict(X)

