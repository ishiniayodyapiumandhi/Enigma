from src.utils import load_sample_data
from src.feature_engineering import FeatureEngineer
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer


def test_pipeline_runs():
    X, y = load_sample_data()
    fe = FeatureEngineer()
    X_eng = fe.transform(X)

    numeric = [c for c in X_eng.columns if X_eng[c].dtype.kind in "fi"]
    categorical = [c for c in X_eng.columns if c not in numeric]

    pre = DataPreprocessor(numeric_features=numeric, categorical_features=categorical)
    X_proc = pre.fit_transform(X_eng)

    trainer = ModelTrainer()
    trainer.build_pipeline(pre, fe)
    trainer.train(X_proc, y)

    preds = trainer.predict(X_proc)
    assert len(preds) == len(y)
