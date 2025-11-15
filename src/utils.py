from typing import Tuple

import pandas as pd


def load_sample_data() -> Tuple[pd.DataFrame, pd.Series]:
	"""Create a tiny synthetic dataset for demo and tests."""
	df = pd.DataFrame({
		"age": [65, 70, 55, 80, 60, 72],
		"education": ["High School", "Bachelor", "High School", "Master", "Bachelor", "High School"],
		"smoker": [0, 1, 0, 0, 1, 0],
		"living_alone": [1, 0, 0, 1, 0, 1],
	})
	# simple synthetic label (not realistic)
	y = pd.Series([1, 1, 0, 1, 0, 1])
	return df, y

