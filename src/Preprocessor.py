from datetime import timedelta

import pandas as pd
from loguru import logger

from data_models.Datasets import Datasets


class Preprocessor:
    def __read_data(self, df_path: str) -> pd.DataFrame:
        return pd.read_csv(df_path)

    def __drop_columns(self, df: pd.DataFrame) -> None:
        df.drop(
            columns=["State.1", "Country", "City", "Year_Month", "Gender"], inplace=True
        )
        logger.debug("OK")

    def __rename_columns(self, df: pd.DataFrame) -> None:
        new_column_names = {
            "Transaction_id": "transaction_id",
            "Date": "date",
            "Product": "product",
            "Device_Type": "device_type",
            "State": "state",
            "Category": "category",
            "Customer_Login_type": "customer_login_type",
            "Delivery_Type": "delivery_type",
            "Transaction_Result": "transaction_result",
            "Amount US$": "amount_us",
            "Individual_Price_US$": "individual_price_us",
            "Time": "time",
            "Quantity": "quantity",
        }
        df.rename(columns=new_column_names, inplace=True)
        logger.debug("OK")

    def __fix_amount_us(self, df: pd.DataFrame) -> pd.DataFrame:
        df["amount_us"] = df["amount_us"].str.replace(",", "")
        df["amount_us"] = df["amount_us"].fillna(0)
        df["amount_us"] = df["amount_us"].astype(int)
        logger.debug("OK")

        return df

    def __fix_individual_price_us(self, df: pd.DataFrame) -> pd.DataFrame:
        def replace_value(row):
            if row["individual_price_us"] == "#VALUE!":
                return row["amount_us"] / row["quantity"]
            else:
                return row["individual_price_us"]

        df["individual_price_us"] = df["individual_price_us"].str.replace(",", "")
        df["individual_price_us"] = df.apply(replace_value, axis=1)
        df["individual_price_us"] = df["individual_price_us"].astype(int)
        logger.debug("OK")

        return df

    def __time_variable_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date_time_str"] = df["date"] + " " + df["time"]
        df["datetime"] = pd.to_datetime(df["date_time_str"], format="%d/%m/%Y %H:%M:%S")
        df = df.sort_values(by="datetime")
        df.drop(columns=["date", "time", "date_time_str"], inplace=True)
        logger.debug("OK")
        return df

    def __one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df["device_type"] = df["device_type"].map({"Web": 0, "Mobile": 1})
        df["delivery_type"] = df["delivery_type"].map(
            {"Normal Delivery": 0, "one-day deliver": 1}
        )
        logger.debug("OK")
        return df

    def __feature_engineering_before_splitting(self, df: pd.DataFrame) -> pd.DataFrame:
        df["days_since_last_order"] = (
            df.groupby("customer_id")["datetime"].diff().dt.days
        )
        df["days_since_last_order"] = df["days_since_last_order"].fillna(0)

        df["is_working_time"] = df["datetime"].dt.hour.apply(
            lambda hour: 1 if 9 <= hour < 18 else 0
        )
        df["is_weekend"] = (df["datetime"].dt.dayofweek >= 5).astype(int)
        df["month"] = df["datetime"].dt.month
        df["year"] = df["datetime"].dt.year
        logger.debug("OK")
        return df

    def __add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = 0
        for i, row in df.iterrows():
            transaction_date = row["datetime"]
            customer_id = row["customer_id"]
            future_transactions: pd.DataFrame = df[
                (df["customer_id"] == customer_id)
                & (df["datetime"] > transaction_date)
                & (df["datetime"] <= transaction_date + timedelta(days=60))
            ]
            if not future_transactions.empty:
                df.loc[i, "target"] = 1
        logger.debug("OK")

        return df

    def __split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        current_datetime = df["datetime"].max()
        test_end = current_datetime - pd.DateOffset(months=2)
        val_end = test_end - pd.DateOffset(weeks=1)
        train_end = val_end - pd.DateOffset(weeks=1)

        train: pd.DataFrame = df[df["datetime"] < train_end].copy()
        val: pd.DataFrame = df[
            (df["datetime"] >= train_end) & (df["datetime"] < val_end)
        ].copy()
        test: pd.DataFrame = df[
            (df["datetime"] >= val_end) & (df["datetime"] < test_end)
        ].copy()
        full_dataset = df.copy()
        logger.debug("OK")
        return train, val, test, full_dataset

    def __target_encoding(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        full_dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def inner_target_encoding(
            column: str,
            new_column_name: str,
            train: pd.DataFrame = train,
            val: pd.DataFrame = val,
            test: pd.DataFrame = test,
            full_dataset: pd.DataFrame = full_dataset,
            target_name: str = "target",
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            category_means = train.groupby(column)[target_name].mean()
            global_mean = train[target_name].mean()

            train[new_column_name] = train[column].map(category_means)
            val[new_column_name] = val[column].map(category_means).fillna(global_mean)
            test[new_column_name] = test[column].map(category_means).fillna(global_mean)
            full_dataset[new_column_name] = (
                full_dataset[column].map(category_means).fillna(global_mean)
            )

            train.drop(columns=column, inplace=True)
            val.drop(columns=column, inplace=True)
            test.drop(columns=column, inplace=True)
            full_dataset.drop(columns=column, inplace=True)

            return train, val, test, full_dataset

        train, val, test, full_dataset = inner_target_encoding(
            "product", "product_encoded"
        )
        train, val, test, full_dataset = inner_target_encoding("state", "state_encoded")
        train, val, test, full_dataset = inner_target_encoding(
            "category", "category_encoded"
        )
        train, val, test, full_dataset = inner_target_encoding(
            "customer_login_type", "customer_login_type_encoded"
        )

        logger.debug("OK")
        return train, val, test, full_dataset

    def __feature_engineering_after_splitting(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        full_dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train["average_days_between_orders"] = train.groupby("customer_id")[
            "days_since_last_order"
        ].transform("mean")
        full_dataset["average_days_between_orders"] = full_dataset.groupby(
            "customer_id"
        )["days_since_last_order"].transform("mean")
        aggs = (
            train.groupby("customer_id")["days_since_last_order"].agg("mean").to_dict()
        )
        val["average_days_between_orders"] = val["customer_id"].map(aggs).fillna(0)
        test["average_days_between_orders"] = test["customer_id"].map(aggs).fillna(0)

        def customer_behavior(
            transform_column: str,
            new_column_name: str,
            func: str,
            groupby_column: str = "customer_id",
            train: pd.DataFrame = train,
            val: pd.DataFrame = val,
            test: pd.DataFrame = test,
            full_dataset: pd.DataFrame = full_dataset,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

            if func == "count":
                train[new_column_name] = train.groupby(groupby_column)[
                    transform_column
                ].transform("count")
                full_dataset[new_column_name] = full_dataset.groupby(groupby_column)[
                    transform_column
                ].transform("count")
            elif func == "mean":
                train[new_column_name] = train.groupby(groupby_column)[
                    transform_column
                ].transform("mean")
                full_dataset[new_column_name] = full_dataset.groupby(groupby_column)[
                    transform_column
                ].transform("mean")

            aggs = train.groupby(groupby_column)[transform_column].agg(func).to_dict()
            val[new_column_name] = val[groupby_column].map(aggs).fillna(0)
            test[new_column_name] = test[groupby_column].map(aggs).fillna(0)

            return train, val, test, full_dataset

        train, val, test, full_dataset = customer_behavior(
            "transaction_id", "customer_num_of_transactions", "count"
        )
        train, val, test, full_dataset = customer_behavior(
            "amount_us", "customer_avg_bill", "mean"
        )
        train, val, test, full_dataset = customer_behavior(
            "individual_price_us", "customer_avg_price_per_item", "mean"
        )
        train, val, test, full_dataset = customer_behavior(
            "quantity", "customer_avg_quantity", "mean"
        )

        logger.debug("OK")
        return train, val, test, full_dataset

    def __drop_datetime(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        full_dataset: pd.DataFrame,
    ) -> None:
        train.drop(columns=["datetime"], inplace=True)
        val.drop(columns=["datetime"], inplace=True)
        test.drop(columns=["datetime"], inplace=True)
        full_dataset.drop(columns=["datetime"], inplace=True)

    def __segregate_target_variable(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        full_dataset: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ]:
        def inner_segregate_target_variable(
            df: pd.DataFrame, target: str = "target"
        ) -> tuple[pd.DataFrame, pd.Series]:
            x = df.drop(target, axis=1)
            y = df[target]
            return x, y

        x_train, y_train = inner_segregate_target_variable(train)
        x_val, y_val = inner_segregate_target_variable(val)
        x_test, y_test = inner_segregate_target_variable(test)
        x_full, y_full = inner_segregate_target_variable(full_dataset)

        logger.debug("OK")
        return x_train, y_train, x_val, y_val, x_test, y_test, x_full, y_full

    def __check_data(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        def inner_check_data(df: pd.DataFrame) -> None:
            assert not df.duplicated().any(), "Error: duplicates detected!"
            assert not df.isnull().values.any(), "Error: Missing values detected!"
            for col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]) or isinstance(
                    df[col], pd.CategoricalDtype
                ), (
                    f"Error: column '{col}' has an invalid data type: {df[col].dtype}. "
                    f"It must be numeric or categorical."
                )

        inner_check_data(train)
        inner_check_data(val)
        inner_check_data(test)
        logger.info("All checks passed successfully.")

    def preprocess(self) -> Datasets:
        df = self.__read_data("data/test_task_data.csv")
        self.__drop_columns(df)
        self.__rename_columns(df)
        df = self.__fix_amount_us(df)
        df = self.__fix_individual_price_us(df)
        df = self.__time_variable_processing(df)
        df = self.__one_hot_encoding(df)
        df = self.__feature_engineering_before_splitting(df)
        df = self.__add_target(df)
        train, val, test, full_dataset = self.__split(df)
        train, val, test, full_dataset = self.__target_encoding(
            train, val, test, full_dataset
        )
        train, val, test, full_dataset = self.__feature_engineering_after_splitting(
            train, val, test, full_dataset
        )
        self.__drop_datetime(train, val, test, full_dataset)
        self.__check_data(train, val, test)
        x_train, y_train, x_val, y_val, x_test, y_test, x_full, y_full = (
            self.__segregate_target_variable(train, val, test, full_dataset)
        )
        logger.debug("OK")

        return Datasets(x_train, y_train, x_val, y_val, x_test, y_test, x_full, y_full)
