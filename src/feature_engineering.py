from typing import Iterable, List

import pandas as pd


class FeatureEngineer:
	"""Minimal feature engineering for allowed non-medical features.

	Example usage:
		fe = FeatureEngineer()
		X = fe.transform(df)
	"""

	def __init__(self):
		pass

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		df = df.copy()
		# Example engineered features
		if "age" in df.columns:
			df["age_sq"] = df["age"] ** 2

		# Simplify education into years if categorical
		if "education" in df.columns and df["education"].dtype == object:
			df["education_level"] = df["education"].apply(self._map_education)

		return df

	@staticmethod
	def _map_education(v: str) -> int:
		if not isinstance(v, str):
			return 0
		v = v.lower()
		if "high" in v:
			return 12
		if "bachelor" in v or "college" in v:
			return 16
		if "master" in v or "ms" in v:
			return 18
		return 10

