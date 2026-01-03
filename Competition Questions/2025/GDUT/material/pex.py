import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import CoolProp.CoolProp as CP

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
file_path = 'data/深部盐水层相关参数 - 副本.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 删除安全系数列
df = df.drop(columns=['安全系数（不可超压）'])

# 处理范围值，提取上下限（不转换为均值）
def split_range(value):
    if isinstance(value, str):
        value = value.replace('–', '-').replace('～', '-').replace('~', '-').strip()
        parts = value.split('-')
        parts = [float(p.strip()) for p in parts]
        return parts
    elif isinstance(value, (int, float)):
        return [float(value), float(value)]
    else:
        return [np.nan, np.nan]

# 提取原始范围数据
df['压力范围_MPa'] = df['压力范围 (MPa)'].apply(split_range)
df['温度范围_C'] = df['温度范围 (°C)'].apply(split_range)

# 处理其他需要均值的列（孔隙度）
def range_to_mean(value):
    if isinstance(value, str):
        value = value.replace('–', '-').replace('～', '-').replace('~', '-').strip()
        parts = value.split('-')
        parts = [float(p.strip()) for p in parts]
        return np.mean(parts)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return np.nan

df['平均孔隙度（%）'] = df['平均孔隙度（%）'].apply(range_to_mean)

# 转换盆地类型为数值（陆上=0，海上=1）
df['盆地类型'] = df['盆地类型'].map({'陆上盆地': 0, '海上盆地': 1})

# 删除固定密度列
df = df.drop(columns=['密度典型值（kg/m³）', '密度范围（假设）', '压力范围 (MPa)', '温度范围 (°C)'])

# 转换盆地面积为平方米（1万平方千米 = 1e10 m²）
df['盆地面积_m2'] = df['盆地面积（万平方千米）'].astype(float) * 1e10

# 定义函数模拟计算密度（使用盐水物性，假设为NaCl溶液）
def calculate_density(pressure_range, temp_range, num_samples=100):
    try:
        p_min, p_max = pressure_range
        t_min, t_max = temp_range
        pressures = np.random.uniform(p_min, p_max, num_samples)
        temps = np.random.uniform(t_min, t_max, num_samples)
        densities = []
        for p, t in zip(pressures, temps):
            # 压力单位转换为Pa，温度转换为K
            density = CP.PropsSI('D', 'P', p*1e6, 'T', t + 273.15, 'water')
            densities.append(density)
        return np.nanmean(densities)
    except Exception as e:
        print(f"计算失败: {e}, 压力范围: {pressure_range}, 温度范围: {temp_range}")
        return np.nan

# 计算每个盆地的平均密度
df['密度_kg/m3'] = df.apply(
    lambda row: calculate_density(row['压力范围_MPa'], row['温度范围_C']), axis=1
)

# 填充NaN值（使用所有盆地密度的均值）
mean_density = df['密度_kg/m3'].mean()
df['密度_kg/m3'] = df['密度_kg/m3'].fillna(mean_density)

# 计算CO₂封存量（孔隙度单位转换为小数）
df['CO2封存量_kg'] = (
    df['盆地面积_m2'] *
    df['平均储层厚度（m）'].astype(float) *
    (df['平均孔隙度（%）'].astype(float)) *  # 转换为小数
    df['密度_kg/m3'] *
    df['S_eff'].astype(float)
)

# 删除冗余列
df = df.drop(columns=['盆地名称', '盆地面积（万平方千米）', '盆地面积_m2', '压力范围_MPa', '温度范围_C', 'S_eff'])

# 计算相关系数矩阵并删除全空列（如果有）
corr_matrix = df.corr()

# 绘制热力图（调整标签角度）
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
    ) 
plt.xticks(rotation=45, ha='right')
plt.title('参数相关性矩阵（含CO2封存量）')
plt.tight_layout()
plt.show()

# 输出相关系数
print("相关系数矩阵：")
print(corr_matrix)