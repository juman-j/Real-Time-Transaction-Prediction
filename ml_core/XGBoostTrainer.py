from ml_core.BaseTrainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("XGBoost")

    def _create_model(self):
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=42, eval_metric="logloss")
