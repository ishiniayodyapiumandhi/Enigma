from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate(y_true: Iterable[int], y_pred: Iterable[int], y_proba: Iterable[float] = None) -> dict:
	res = {}
	res["accuracy"] = float(accuracy_score(y_true, y_pred))
	if y_proba is not None:
		try:
			res["roc_auc"] = float(roc_auc_score(y_true, y_proba))
		except Exception:
			res["roc_auc"] = None
	return res

