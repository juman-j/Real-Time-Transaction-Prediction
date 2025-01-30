from abc import ABC, abstractmethod

from loguru import logger
from pandas import DataFrame, Series, concat
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

from configurations.config import config
from data_models.Datasets import Datasets


class BaseTrainer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.param_grid = getattr(config.model_parameters, model_name).__dict__

    @abstractmethod
    def _create_model(self):
        pass

    def __grid_search(self, model, data: Datasets):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        grid_search.fit(data.x_train, data.y_train)
        logger.info(
            f"Best parameters for {self.model_name}: {grid_search.best_params_}"
        )
        return grid_search.best_estimator_

    def __evaluate(self, model, x: DataFrame, y: Series):
        y_pred = model.predict_proba(x)[:, 1]
        auc = roc_auc_score(y, y_pred)
        logloss = log_loss(y, y_pred)
        return {"roc_auc": auc, "logloss": logloss}

    def train(self, data: Datasets):
        model = self._create_model()
        best_estimator = self.__grid_search(model, data)
        return best_estimator, self.__evaluate(best_estimator, data.x_val, data.y_val)

    def fit_best_model(self, best_model, data: Datasets):
        x_train_val = concat([data.x_train, data.x_val])
        y_train_val = concat([data.y_train, data.y_val])

        best_model.fit(x_train_val, y_train_val)

        test_metrics = self.__evaluate(best_model, data.x_test, data.y_test)
        logger.info(f"Metrics on test data ({self.model_name}): {test_metrics}")

        best_model.fit(data.x_full, data.y_full)
        logger.info(f"Model {self.model_name} trained on the full dataset.")

        return best_model, test_metrics
