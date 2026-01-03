import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
df = pd.read_excel('./data/data-total.xlsx')

# 自定义函数：将孕周字符串转换为数值
def convert_gestational_week(week_str):
    if pd.isna(week_str):
        return None
    match = re.match(r'(\d+)w(?:\+(\d+))?', str(week_str))  # 匹配“周数w+天数”或“周数w”
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return weeks + days / 7.0  # 天数转周数后与周数相加
    return None

# 新增“检测孕周数值”列（存储转换后的孕周）
df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)

# 选择分析列（母序列：Y染色体浓度；比较序列：孕妇BMI、年龄、检测孕周数值、比对比例）
selected_cols = [
    'Y染色体浓度', 
    '孕妇BMI', 
    '年龄',
    '检测孕周数值', 
    '在参考基因组上比对的比例'
]
df_gray = df[selected_cols].copy()
df_gray.dropna(inplace=True)  # 删除缺失值行

# 母序列：Y染色体浓度（reshape为二维数组，便于后续广播计算）
mother_series = df_gray['Y染色体浓度'].values.reshape(-1, 1)  
# 比较序列：其余4个指标（每行1个样本，每列1个指标）
comparison_series = df_gray.iloc[:, 1:].values  

def mean_normalization(data):
    """均值化无量纲化：每个值 ÷ 该指标的均值"""
    mean_vals = np.mean(data, axis=0)
    return data / mean_vals

# 母序列与比较序列均进行均值化
mother_norm = mean_normalization(mother_series)
comparison_norm = mean_normalization(comparison_series)

abs_diff = np.abs(comparison_norm - mother_norm)  # 母序列与各比较序列的逐元素绝对差

rho = 0.5  # 分辨系数
max_diff = np.max(abs_diff)  # 所有绝对差的最大值
min_diff = np.min(abs_diff)  # 所有绝对差的最小值

# 计算关联系数
correlation_coeff = (min_diff + rho * max_diff) / (abs_diff + rho * max_diff)

# 按列求均值（每列对应一个比较序列的关联系数均值）
correlation_degree = np.mean(correlation_coeff, axis=0)  

# 关联度与指标名称匹配，并按关联度降序排序
indices_names = selected_cols[1:]  # 比较序列的指标名称
degree_df = pd.DataFrame({
    '指标': indices_names,
    '灰色关联度': correlation_degree
})
degree_df_sorted = degree_df.sort_values(by='灰色关联度', ascending=False)

# 打印关联度排序结果
print("各指标与Y染色体浓度的灰色关联度（从高到低排序）：")
print(degree_df_sorted)

# 可视化：关联度条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='灰色关联度', y='指标', data=degree_df_sorted, orient='h')
plt.title('各指标与Y染色体浓度的灰色关联度')
plt.xlabel('灰色关联度')
plt.ylabel('指标')
plt.tight_layout()
plt.savefig('gray_correlation_degree.png',dpi=400)
plt.show()