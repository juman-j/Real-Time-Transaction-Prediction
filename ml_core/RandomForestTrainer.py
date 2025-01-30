from ml_core.BaseTrainer import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("RandomForest")

    def _create_model(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=42)
