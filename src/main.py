import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from loguru import logger

from ml_core.RandomForestTrainer import RandomForestTrainer
from ml_core.XGBoostTrainer import XGBoostTrainer
from src.Preprocessor import Preprocessor

if __name__ == "__main__":
    data = Preprocessor().preprocess()

    xgb_trainer = XGBoostTrainer()
    rf_trainer = RandomForestTrainer()

    xgb_model, xgb_metrics = xgb_trainer.train(data)
    rf_model, rf_metrics = rf_trainer.train(data)

    if xgb_metrics.get("roc_auc") > rf_metrics.get("roc_auc"):
        xgb_model, xgb_test_metrics = xgb_trainer.fit_best_model(xgb_model, data)
    else:
        rf_model, rf_test_metrics = rf_trainer.fit_best_model(rf_model, data)

    logger.success("All done.")
