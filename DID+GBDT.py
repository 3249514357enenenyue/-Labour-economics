import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def load_and_enhance_data():
    df = pd.read_excel("14-23log.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
    df['date'] = pd.to_datetime(df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01'), errors='coerce')

    df = df[(df['date'] >= '2014-12-01') & df['WorkType'].isin(['Full-time', 'Part-time']) & df['HourlyPay'].notna()]

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['HOURPAY_lag1'] = df.groupby('WorkType')['HourlyPay'].transform(lambda x: x.shift(1).bfill())
    df['HOURPAY_ma3'] = df.groupby('WorkType')['HourlyPay'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    policy_date = pd.to_datetime('2020-04-01')
    df['post_policy'] = (df['date'] >= policy_date).astype(int)

    return df


def build_preprocessor():
    categorical_features = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome'
    ]
    numerical_features = [
        'year', 'month', 'quarter', 'month_sin', 'month_cos',
        'HOURPAY_lag1', 'HOURPAY_ma3', 'post_policy'
    ]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            sparse_output=False,
            drop='if_binary'
        ), categorical_features)
    ], remainder='drop')

    return preprocessor, categorical_features, numerical_features


def train_and_evaluate(sub_df, label):
    print(f"\n🚀 训练 {label} 模型 (GBDT模型)")

    feature_columns = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector', 'EduLevel',
        'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome',
        'year', 'month', 'quarter', 'month_sin', 'month_cos',
        'HOURPAY_lag1', 'HOURPAY_ma3', 'post_policy'
    ]

    X = sub_df[feature_columns]
    y = sub_df['HourlyPay']

    categorical_cols = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome'
    ]
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    preprocessor, categorical_features, numerical_features = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # 优化后的GBDT参数
    model = GradientBoostingRegressor(
        n_estimators=200,  # 减少树的数量，配合更小的学习率
        learning_rate=0.1,  # 适当增大学习率
        max_depth=5,  # 减小树的最大深度
        min_samples_split=20,  # 增加分裂所需最小样本数
        min_samples_leaf=10,  # 增加叶节点最小样本数
        max_features='sqrt',  # 考虑使用特征平方根数量
        subsample=0.8,  # 子采样比例
        loss='huber',  # 使用Huber损失函数增强鲁棒性
        random_state=42,
        validation_fraction=0.1,  # 验证集比例用于早停
        n_iter_no_change=10,  # 早停轮数
        tol=1e-4  # 早停容忍度
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n📊 模型评估指标:")
    print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}  MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")

    # 计算政策效应（对比 post_policy 0 和 1 的预测值）
    did_scenarios = pd.DataFrame({'post_policy': [0, 1]})
    for col in X.columns:
        if col not in did_scenarios:
            if col in categorical_features:
                did_scenarios[col] = X[col].mode()[0]
            else:
                did_scenarios[col] = X[col].median()

    for col in categorical_features:
        if col in did_scenarios:
            did_scenarios[col] = did_scenarios[col].astype(str)

    X_transformed_scenarios = preprocessor.transform(did_scenarios)
    predictions = model.predict(X_transformed_scenarios)

    effect_size = predictions[1] - predictions[0]  # 政策后 - 政策前

    print(f"\n📌 {label} 政策效应: {effect_size:.2f}")

    # SHAP值分析
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:100])  # 计算前100个样本的SHAP值

    # 绘制SHAP摘要图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.title(f"{label} - SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{label}.png", dpi=300)
    plt.close()

    return {
        'model': model,
        'preprocessor': preprocessor,
        'effect_size': effect_size,
        'metrics': {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        },
        'shap_values': shap_values
    }


def run_gbdt_analysis(df):
    results = {}

    # 分割数据集
    df_full_time = df[df['WorkType'] == 'Full-time'].copy()
    df_part_time = df[df['WorkType'] == 'Part-time'].copy()

    results['Full-time'] = train_and_evaluate(df_full_time, "Full-time")
    results['Part-time'] = train_and_evaluate(df_part_time, "Part-time")

    print("\n🚀 **对比 Full-time 与 Part-time**")
    print(f"Full-time 政策效应: {results['Full-time']['effect_size']:.2f}")
    print(f"Part-time 政策效应: {results['Part-time']['effect_size']:.2f}")
    print(
        f"二者差值 (Full-time - Part-time): {(results['Full-time']['effect_size'] - results['Part-time']['effect_size']):.2f}")

    # 可视化对比
    plt.figure(figsize=(10, 6))
    plt.bar(['Full-time', 'Part-time'],
            [results['Full-time']['effect_size'], results['Part-time']['effect_size']],
            color=['blue', 'orange'])
    plt.title("Policy Effect Comparison (GBDT)")
    plt.ylabel("Effect Size")
    plt.savefig("policy_effect_comparison_gbdt.png", dpi=300)
    plt.close()

    return results


def main():
    df = load_and_enhance_data()
    results = run_gbdt_analysis(df)


if __name__ == "__main__":
    main()