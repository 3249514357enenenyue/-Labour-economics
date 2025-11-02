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


def train_group(group_df, group_name):
    # Split data into pre-policy and post-policy
    pre_policy_df = group_df[group_df['date'] < '2020-04-01'].copy()
    post_policy_df = group_df[group_df['date'] >= '2020-04-01'].copy()

    # Find the earliest available month after policy change
    if not post_policy_df.empty:
        first_post_month = post_policy_df['date'].dt.to_period('M').min()
        first_month_post = post_policy_df[
            post_policy_df['date'].dt.to_period('M') == first_post_month
            ].copy()
    else:
        raise ValueError(f"No post-policy data available for {group_name} group")

    if pre_policy_df.empty:
        raise ValueError(f"{group_name} group pre-policy data is empty")
    if first_month_post.empty:
        raise ValueError(f"No data available for first post-policy month ({first_post_month}) for {group_name} group")

    print(f"\n{'-' * 30}\n{group_name} group pre-processing dimensions - Pre-policy: {pre_policy_df.shape}")
    print(f"First post-policy month: {first_post_month}, data points: {first_month_post.shape[0]}")

    # Feature selection
    feature_columns = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome',
        'year', 'month', 'quarter', 'month_sin', 'month_cos',
        'HOURPAY_lag1', 'HOURPAY_ma3'
    ]
    processed_pre = pre_policy_df[feature_columns].copy()
    processed_post = first_month_post[feature_columns].copy()

    # Convert categorical features to string
    categorical_features = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome'
    ]
    processed_pre[categorical_features] = processed_pre[categorical_features].astype(str)
    processed_post[categorical_features] = processed_post[categorical_features].astype(str)

    # Preprocessing
    preprocessor = build_preprocessor()
    try:
        X_train = preprocessor.fit_transform(processed_pre)
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.toarray()
        X_train = X_train.astype(np.float32)

        X_test = preprocessor.transform(processed_post)
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.toarray()
        X_test = X_test.astype(np.float32)

        print("Post-processing data type:", X_train.dtype, "Contains NaN:", np.isnan(X_train).any())
    except Exception as e:
        print(f"Preprocessing failed. Last 5 rows:\n{processed_pre.tail()}")
        raise

    # Target processing
    y_train = pre_policy_df['HourlyPay'].values.reshape(-1, 1)
    y_test = first_month_post['HourlyPay'].values.reshape(-1, 1)

    y_scaler = QuantileTransformer(output_distribution='normal')
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # GBDT model configuration
    gbdt = GradientBoostingRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        min_samples_split=30,
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=20
    )

    # Train model
    gbdt.fit(X_train, y_train_scaled.ravel())

    # Predict
    y_pred_scaled = gbdt.predict(X_test).reshape(-1, 1)

    # Inverse transform
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_orig = y_scaler.inverse_transform(y_test_scaled)

    # Evaluation metrics
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    print(f"\n{group_name} Evaluation Results for {first_post_month}:")
    print(f"MSE: {mse:.4f}  RMSE: {np.sqrt(mse):.4f}  R²: {r2:.4f}")

    # Visualization
    plt.figure(figsize=(15, 5))

    # Actual vs Predicted comparison
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_orig, y_pred, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('Actual Hourly Pay')
    plt.ylabel('Predicted Hourly Pay')
    plt.title(f'{group_name} Actual vs Predicted ({first_post_month})')

    # Distribution comparison
    plt.subplot(1, 2, 2)
    plt.hist(y_test_orig, bins=30, alpha=0.5, label='Actual')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted')
    plt.xlabel('Hourly Pay')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'{group_name} Distribution Comparison ({first_post_month})')

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
        models[group], preprocessors[group], y_scalers[group] = train_group(group_df, group)


if __name__ == "__main__":
    main()