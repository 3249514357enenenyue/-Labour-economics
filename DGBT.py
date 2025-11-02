import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_and_enhance_data():
    df = pd.read_excel("14-23log.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])

    df['date'] = pd.to_datetime(
        df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01'),
        errors='coerce'
    )

    mask = (
            (df['date'] >= '2014-12-01') &
            (df['WorkType'].isin(['Full-time', 'Part-time'])) &
            (df['HourlyPay'].notna())
    )
    df = df[mask].copy()

    if df.empty:
        raise ValueError("Data is empty after filtering")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['HOURPAY_lag1'] = df.groupby('WorkType')['HourlyPay'].transform(
        lambda x: x.shift(1).bfill()
    )
    df['HOURPAY_ma3'] = df.groupby('WorkType')['HourlyPay'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    required_columns = ['year', 'month', 'quarter', 'month_sin', 'month_cos',
                        'HOURPAY_lag1', 'HOURPAY_ma3', 'WorkType', 'HourlyPay']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required feature columns: {missing}")

    return df


def build_preprocessor():
    categorical_features = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome'
    ]
    numerical_features = [
        'year', 'month', 'quarter', 'month_sin', 'month_cos',
        'HOURPAY_lag1', 'HOURPAY_ma3'
    ]


    return ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            sparse_output=False,
            drop='if_binary'
        ), categorical_features)
    ], remainder='drop')


def split_data(pre_policy_df, test_size=0.1, val_size=0.2):
    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        pre_policy_df,
        test_size=test_size,
        random_state=42
    )

    # Second split: separate train and validation from remaining data
    # Adjust val_size to account for already removed test data
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=42
    )

    return train_df, val_df, test_df


def train_and_evaluate_group(group_df, group_name):
    # Filter data before April 2020
    pre_policy_df = group_df[group_df['date'] < '2020-04-01'].copy()

    if pre_policy_df.empty:
        raise ValueError(f"{group_name} group pre-policy data is empty")

    # Split data into train (70%), validation (20%), and test (10%)
    train_df, val_df, test_df = split_data(pre_policy_df)

    print(f"\n{'-' * 30}\n{group_name} group data split:")
    print(f"Training set: {len(train_df)} samples ({len(train_df) / len(pre_policy_df):.1%})")
    print(f"Validation set: {len(val_df)} samples ({len(val_df) / len(pre_policy_df):.1%})")
    print(f"Test set: {len(test_df)} samples ({len(test_df) / len(pre_policy_df):.1%})")

    # Feature selection
    feature_columns = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome',
        'year', 'month', 'quarter', 'month_sin', 'month_cos',
        'HOURPAY_lag1', 'HOURPAY_ma3'
    ]

    # Convert categorical features to string
    categorical_features = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome'
    ]
    train_df[feature_columns] = train_df[feature_columns].copy()
    val_df[feature_columns] = val_df[feature_columns].copy()
    test_df[feature_columns] = test_df[feature_columns].copy()

    train_df[categorical_features] = train_df[categorical_features].astype(str)
    val_df[categorical_features] = val_df[categorical_features].astype(str)
    test_df[categorical_features] = test_df[categorical_features].astype(str)

    # Preprocessing
    preprocessor = build_preprocessor()
    try:
        # Fit on training data only
        X_train = preprocessor.fit_transform(train_df[feature_columns])
        X_val = preprocessor.transform(val_df[feature_columns])
        X_test = preprocessor.transform(test_df[feature_columns])

        if not isinstance(X_train, np.ndarray):
            X_train = X_train.toarray()
            X_val = X_val.toarray()
            X_test = X_test.toarray()

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        print("\nPost-processing data shapes:")
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    except Exception as e:
        print(f"Preprocessing failed. Last 5 rows:\n{train_df[feature_columns].tail()}")
        raise

    # Target processing
    y_train = train_df['HourlyPay'].values.reshape(-1, 1)
    y_val = val_df['HourlyPay'].values.reshape(-1, 1)
    y_test = test_df['HourlyPay'].values.reshape(-1, 1)

    y_scaler = QuantileTransformer(output_distribution='normal')
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    # GBDT model configuration
    gbdt = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        min_samples_split=30,
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=20
    )

    # Train model
    print("\nTraining model...")
    gbdt.fit(X_train, y_train_scaled.ravel())

    # Evaluate on validation set
    y_val_pred_scaled = gbdt.predict(X_val).reshape(-1, 1)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
    y_val_orig = y_scaler.inverse_transform(y_val_scaled)

    val_mse = mean_squared_error(y_val_orig, y_val_pred)
    val_r2 = r2_score(y_val_orig, y_val_pred)
    print(f"\nValidation Results for {group_name}:")
    print(f"MSE: {val_mse:.4f}  RMSE: {np.sqrt(val_mse):.4f}  R²: {val_r2:.4f}")

    # Evaluate on test set
    y_test_pred_scaled = gbdt.predict(X_test).reshape(-1, 1)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)
    y_test_orig = y_scaler.inverse_transform(y_test_scaled)

    test_mse = mean_squared_error(y_test_orig, y_test_pred)
    test_r2 = r2_score(y_test_orig, y_test_pred)
    print(f"\nTest Results for {group_name}:")
    print(f"MSE: {test_mse:.4f}  RMSE: {np.sqrt(test_mse):.4f}  R²: {test_r2:.4f}")

    # Visualization
    plt.figure(figsize=(18, 6))

    # Actual vs Predicted comparison (Test set)
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_orig, y_test_pred, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('Actual Hourly Pay')
    plt.ylabel('Predicted Hourly Pay')
    plt.title(f'{group_name} Test Set: Actual vs Predicted')

    # Distribution comparison (Test set)
    plt.subplot(1, 3, 2)
    plt.hist(y_test_orig, bins=30, alpha=0.5, label='Actual')
    plt.hist(y_test_pred, bins=30, alpha=0.5, label='Predicted')
    plt.xlabel('Hourly Pay')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'{group_name} Test Set: Distribution Comparison')

    # Feature importance
    plt.subplot(1, 3, 3)
    feature_importance = gbdt.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.barh(pos, feature_importance[sorted_idx], align='center', height=0.8)
    feature_names = preprocessor.get_feature_names_out()
    plt.yticks(pos, np.array(feature_names)[sorted_idx], fontsize=8)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Ranking')

    plt.tight_layout()
    plt.show()

    return gbdt, preprocessor, y_scaler


def main():
    df = load_and_enhance_data()

    print("\nData Quality Report:")
    print(f"Total samples: {len(df)}")
    print(f"Full-time samples: {len(df[df.WorkType == 'Full-time'])}")
    print(f"Part-time samples: {len(df[df.WorkType == 'Part-time'])}")
    print("\nKey Feature Statistics:")
    print(df[['HourlyPay', 'HOURPAY_lag1', 'HOURPAY_ma3']].describe().round(2))

    models = {}
    preprocessors = {}
    y_scalers = {}
    for group in ['Full-time', 'Part-time']:
        group_df = df[df.WorkType == group].copy()
        if group_df.empty:
            print(f"\nWarning: {group} group has no data")
            continue

        print(f"\n{'=' * 30}\nProcessing {group} group\n{'=' * 30}")
        models[group], preprocessors[group], y_scalers[group] = train_and_evaluate_group(group_df, group)


if __name__ == "__main__":
    main()