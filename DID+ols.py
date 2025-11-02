import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


# 加载数据并添加 DID 相关变量
def load_and_enhance_data():
    df = pd.read_excel("14-23xin.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
    print("\n✅ 数据加载成功！")
    print(f"🔹 数据规模: {df.shape}")
    print(f"🔹 列名: {list(df.columns)}")

    df['date'] = df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01', regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"\n📆 数据时间范围: {df['date'].min().date()} ➝ {df['date'].max().date()}")

    df = df[df['WorkType'].isin(['Full-time', 'Part-time'])].copy()
    print(f"🔹 过滤无效 WorkType 后，数据规模: {df.shape}")

    df = df[df['HourlyPay'].notna()].copy()
    print(f"🔹 过滤 HourlyPay 缺失值后，数据规模: {df.shape}")

    if df.empty:
        raise ValueError("❌ 过滤后数据为空，请检查数据源！")

    policy_date = pd.to_datetime('2020-04-01')

    # 添加政策是否生效变量
    df['post_policy'] = (df['date'] >= policy_date).astype(int)

    return df


# 计算模型的评价指标
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # 避免除以 0
    r2 = r2_score(y_true, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}


# 运行 DID 分析（分别建模）
def run_separate_did_analysis(df):
    results = {}

    for work_type in ["Full-time", "Part-time"]:
        print(f"\n🔍 {work_type} 组的 DID 回归分析:\n")

        sub_df = df[df['WorkType'] == work_type].copy()

        # 单独建模
        model = smf.ols("HourlyPay ~ post_policy", data=sub_df).fit()
        print(model.summary())

        # 计算预测值
        y_true = sub_df['HourlyPay']
        y_pred = model.predict(sub_df)

        # 计算回归评价指标
        metrics = evaluate_model(y_true, y_pred)
        print(f"\n📊 {work_type} 模型评估指标：")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # 提取政策影响的 DID 估计
        did_effect = model.params['post_policy']
        did_ci = model.conf_int().loc['post_policy']

        print(f"\n📌 **{work_type} 组的政策效应 (ATT):** {did_effect:.4f} (95% CI: {did_ci[0]:.4f} - {did_ci[1]:.4f})")

        results[work_type] = {"ATT": did_effect, "CI": did_ci}

        # 可视化政策影响
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Before Policy', 'After Policy'], y=[sub_df[sub_df['post_policy'] == 0]['HourlyPay'].mean(),
                                                            sub_df[sub_df['post_policy'] == 1]['HourlyPay'].mean()],
                    palette=['blue', 'red'])
        plt.title(f"Hourly Pay Before and After Policy: {work_type}")
        plt.ylabel("Hourly Pay (£)")
        plt.show()

    return results


# 主函数
def main():
    df = load_and_enhance_data()

    print("\n📊 数据质量报告:")
    print(f"🔹 总样本数: {len(df)}")
    print(f"🔹 Full-time 样本数: {len(df[df['WorkType'] == 'Full-time'])}")
    print(f"🔹 Part-time 样本数: {len(df[df['WorkType'] == 'Part-time'])}")
    print("\n📌 关键特征统计:\n", df[['HourlyPay', 'post_policy']].describe().round(2))

    print("\n📊 分别运行 Full-time 和 Part-time 的 DID 分析")
    did_results = run_separate_did_analysis(df)

    print("\n✅ **最终 DID 估计的政策效应 (ATT)（分别建模）:**")
    for work_type, result in did_results.items():
        print(f"  {work_type}: {result['ATT']:.4f} (95% CI: {result['CI'][0]:.4f} - {result['CI'][1]:.4f})")


if __name__ == "__main__":
    main()
