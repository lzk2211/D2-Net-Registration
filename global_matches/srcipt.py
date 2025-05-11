import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('global_matches.csv')  # 将 'your_file.csv' 替换为你的实际文件名

# 坐标转换
df['GlobalLeftX'] = round(df['GlobalLeftX'] / 30720 * 2600).astype(int)
df['GlobalRightX'] = round(df['GlobalRightX'] / 30720 * 2600).astype(int)
df['GlobalLeftY'] = round(df['GlobalLeftY'] / 17408 * 1476).astype(int)
df['GlobalRightY'] = round(df['GlobalRightY'] / 17408 * 1476).astype(int)

df = df[(df['GlobalLeftX'] <= 1476) & (df['GlobalRightX'] <= 1476)]

# 将结果保存为新的 CSV 文件
df.to_csv('converted_result_1.csv', index=False)

print("转换完成，结果已保存为 converted_result_1.csv")

