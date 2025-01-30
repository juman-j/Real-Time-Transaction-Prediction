from dataclasses import dataclass

from pandas import DataFrame, Series


@dataclass
class Datasets:
    x_train: DataFrame
    y_train: Series
    x_val: DataFrame
    y_val: Series
    x_test: DataFrame
    y_test: Series
    x_full: DataFrame
    y_full: Series
