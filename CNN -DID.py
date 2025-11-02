import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rdrobust import rdrobust, rdplot

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_and_enhance_data():
    df = pd.read_excel("14-23.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
    df['date'] = pd.to_datetime(df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01'), errors='coerce')

    mask = (df['date'] >= '2014-12-01') & (df['FTPTWK'].isin(['Full-time', 'Part-time'])) & (df['HOURPAY'].notna())
    df = df[mask].copy()

    if df.empty:
        raise ValueError("Data is empty after filtering")

    df['time_to_treatment'] = (df['date'] - pd.to_datetime('2020-04-01')).dt.days / 30
    df['treated'] = (df['date'] >= '2020-04-01').astype(int)
    df['running_var'] = df['time_to_treatment']

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['HOURPAY_lag1'] = df.groupby('FTPTWK')['HOURPAY'].transform(lambda x: x.shift(1).bfill())
    df['HOURPAY_ma3'] = df.groupby('FTPTWK')['HOURPAY'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    return df


def build_preprocessor():
    categorical_features = ['SEX', 'Inde07m', 'MARSTA', 'REGWKR', 'SECTOR', 'YPTJOB', 'ETUKEUL', 'BENFTS', 'LESPAY',
                            'HOME']
    numerical_features = ['year', 'month', 'quarter', 'month_sin', 'month_cos', 'HOURPAY_lag1', 'HOURPAY_ma3',
                          'running_var']

    return ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, drop='if_binary'),
         categorical_features)
    ], remainder='drop')


def perform_rdd_analysis(df, group_name):
    rdd_df = df[(df['time_to_treatment'] >= -24) & (df['time_to_treatment'] <= 24)].copy()
    if rdd_df.empty:
        print(f"Not enough data for RDD analysis in {group_name} group")
        return None

    y = rdd_df['HOURPAY'].values
    x = rdd_df['running_var'].values
    c = 0

    print("\nCalculating optimal bandwidth...")
    try:
        bw = rdrobust(y, x, c=c)
        optimal_bw = bw.bws[0, 0]
        print(f"Optimal bandwidth: {optimal_bw:.2f} months")
    except Exception as e:
        print(f"Error in bandwidth selection: {e}")
        optimal_bw = 12

    print("\nGenerating RD plot...")
    plt.figure(figsize=(10, 6))
    rdplot(y, x, c=c, h=optimal_bw, p=1, nbins=50,
           title=f'RD Plot for {group_name} Group',
           x_label='Months Relative to Policy Change',
           y_label='Hourly Pay')
    plt.show()

    return optimal_bw


def train_group(group_df, group_name):
    bandwidth = perform_rdd_analysis(group_df, group_name)

    pre_policy_df = group_df[group_df['date'] < '2020-04-01'].copy()
    post_policy_df = group_df[group_df['date'] >= '2020-04-01'].copy()

    if post_policy_df.empty:
        raise ValueError(f"No post-policy data available for {group_name} group")

    first_post_month = post_policy_df['date'].dt.to_period('M').min()
    first_month_post = post_policy_df[post_policy_df['date'].dt.to_period('M') == first_post_month].copy()

    if pre_policy_df.empty:
        raise ValueError(f"{group_name} group pre-policy data is empty")
    if first_month_post.empty:
        raise ValueError(f"No data available for first post-policy month ({first_post_month}) for {group_name} group")

    print(f"\nProcessing {group_name} group - Pre-policy: {pre_policy_df.shape}")

    feature_columns = ['SEX', 'Inde07m', 'MARSTA', 'REGWKR', 'SECTOR', 'YPTJOB', 'ETUKEUL', 'BENFTS', 'LESPAY', 'HOME',
                       'year', 'month', 'quarter', 'month_sin', 'month_cos', 'HOURPAY_lag1', 'HOURPAY_ma3',
                       'running_var']

    processed_pre = pre_policy_df[feature_columns].copy()
    processed_post = first_month_post[feature_columns].copy()
    categorical_features = ['SEX', 'Inde07m', 'MARSTA', 'REGWKR', 'SECTOR', 'YPTJOB', 'ETUKEUL', 'BENFTS', 'LESPAY',
                            'HOME']

    processed_pre[categorical_features] = processed_pre[categorical_features].astype(str)
    processed_post[categorical_features] = processed_post[categorical_features].astype(str)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(processed_pre)
    X_test = preprocessor.transform(processed_post)

    y_train = pre_policy_df['HOURPAY'].values.reshape(-1, 1)
    y_test = first_month_post['HOURPAY'].values.reshape(-1, 1)

    y_scaler = QuantileTransformer(output_distribution='normal')
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    gbdt = GradientBoostingRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                                     min_samples_split=30, random_state=42)
    gbdt.fit(X_train, y_train_scaled.ravel())

    y_pred_scaled = gbdt.predict(X_test).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(
        f"{group_name} Evaluation Results for {first_post_month}: MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")

    return gbdt, preprocessor, y_scaler


def main():
    df = load_and_enhance_data()
    models = {}
    preprocessors = {}
    y_scalers = {}

    for group in ['Full-time', 'Part-time']:
        group_df = df[df.FTPTWK == group].copy()
        if group_df.empty:
            print(f"Warning: {group} group has no data")
            continue

        models[group], preprocessors[group], y_scalers[group] = train_group(group_df, group)


if __name__ == "__main__":
    main()
