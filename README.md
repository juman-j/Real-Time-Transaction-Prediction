# Real Time Transaction Prediction

## Project Structure

*   **EDA_and_Modeling.ipynb:** This Jupyter Notebook contains the complete data analysis process:
    *   Exploratory Data Analysis (EDA)
    *   Feature Engineering
    *   Model Training and Evaluation (Binary Classification)

*   **src/**:

    *   **main.py:** Runs the entire pipeline.
    *   **Preprocessor.py:** Pipeline for processing data, generating features and passing checks.

*   **ml_core/**:

    *   **BaseTrainer.py:** Basic code for model training.
    *   **RandomForestTrainer.py:** RandomForest realization.
    *   **XGBoostTrainer.py:** RandomForest realization.

*   **data_models/**:
    *   **config_models.py:** dataclass for storing hyperparameters and other configuration values.
    *   **Datasets.py:** dataclass for storing data.


## Dataset

The dataset should be in CSV format with the following columns:

*   `transaction_id`: Unique transaction identifier.
*   `customer_id`: Unique customer identifier.
*   `device_type`: Device used for the transaction.
*   `delivery_type`: Delivery method.
*   `transaction_result`: Transaction result.
*   `amount_us`: Transaction amount (USD).
*   `individual_price_us`: Price per item (USD).
*   `quantity`: Quantity of items.
*   `datetime`: Transaction date and time.
*   `year`: Year of the transaction.
*   `month`: Month of the transaction.
*   `days_since_last_order`: Days since the customer's last order.
*   `is_working_time`: 0/1 indicating if the transaction occurred during working hours.
*   `is_weekend`: 0/1 indicating if the transaction occurred on a weekend.
*   `product_encoded`: Encoded product.
*   `state_encoded`: Encoded state.
*   `category_encoded`: Encoded product category.
*   `customer_login_type_encoded`: Encoded customer login type.
*   `average_days_between_orders`: Average number of days between customer orders.
*   `customer_num_of_transactions`: Total number of transactions for the customer.
*   `customer_avg_bill`: Average customer bill amount.
*   `customer_avg_price_per_item`: Average price per item for the customer.
*   `customer_avg_quantity`: Average quantity of items per order for the customer.
*   `target`: Target variable (1 if a subsequent transaction occurred within 2 months, 0 otherwise).

## Libraries Used

*   pandas
*   numpy
*   scikit-learn
*   xgboost
*   matplotlib
*   seaborn
*   loguru

## Model Evaluation Metrics

*   AUC-ROC
*   Log-loss
