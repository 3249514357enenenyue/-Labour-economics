import os
import pandas as pd
from collections import defaultdict

folder_path = "C:\\Users\\DELL\\Desktop\\14年4季度-23年2季度\\new"  # 注意路径中的反斜杠需要转义
all_files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.xlsx', '.xls'))]

all_dfs = []
all_columns = []  # 存储每个文件的列信息
column_records = defaultdict(list)  # 记录每个列出现的文件

# 读取文件并记录列信息
for file in all_files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_excel(file_path)
        # 记录当前文件的列
        cols = list(df.columns)
        all_columns.append(set(cols))
        all_dfs.append(df)
        # 记录列出现情况
        for col in cols:
            column_records[col].append(file)
    except Exception as e:
        print(f"⚠️ 读取文件 {file} 时出错: {str(e)}")

# 检测不一致列
inconsistent_cols = []
for col, files in column_records.items():
    if len(files) != len(all_files):
        inconsistent_cols.append(col)

# 生成警告信息
if inconsistent_cols:
    print("\n⚠️ 警告：发现不一致的列！")
    print("以下列未在所有文件中出现：")

    for col in sorted(inconsistent_cols):
        exist_files = column_records[col]
        missing_files = [f for f in all_files if f not in exist_files]

        print(f"\n■ 列名：'{col}'")
        print(f"  存在于：{len(exist_files)} 个文件（示例：{exist_files[:3]}）")
        print(f"  缺失于：{len(missing_files)} 个文件（示例：{missing_files[:3]}）")

# 合并所有数据
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 替换 -9 为 "No Answer"
    combined_df.replace([-9, "-9"], "No Answer", inplace=True)

    # 保存为CSV（解决内存问题）
    output_path = os.path.join(folder_path, 'combined_output.csv')
    try:
        # 使用utf-8-sig编码解决中文乱码问题
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n✅ 合并完成！共处理 {len(all_files)} 个文件")
        print(f"合并后包含 {len(combined_df)} 行，{len(combined_df.columns)} 列")
        print(f"文件已保存到：{output_path}")

        # 最终列差异提醒
        if inconsistent_cols:
            print("\n注意：合并后的文件包含以下特殊列：")
            print([col for col in combined_df.columns if col in inconsistent_cols])

    except Exception as e:
        print(f"\n❌ 保存文件时出错: {str(e)}")
else:
    print("\n⚠️ 没有找到可合并的数据！请检查输入文件。")
