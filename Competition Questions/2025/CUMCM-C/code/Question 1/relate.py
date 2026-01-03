import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import re

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def convert_gestational_week(week_str):
    """
    将孕周字符串（如'13w+6'、'14w'）转换为数值型总周数（保留3位小数）
    """
    if pd.isna(week_str):  # 处理缺失值
        return None
    # 正则匹配“周数w+天数”或“周数w”格式
    match = re.match(r'(\d+)w(?:\+(\d+))?', week_str)
    if match:
        weeks = int(match.group(1))  # 提取周数
        days = int(match.group(2)) if match.group(2) else 0  # 提取天数（无则为0）
        total_weeks = weeks + days / 7.0  # 转换为总周数（天数/7转为周）
        return round(total_weeks, 3)
    else:
        return None

# 1. 读取Excel数据
df = pd.read_excel('./data/data-total.xlsx')

# 2. 转换“检测周”为数值型孕周
df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)

# 3. 选择分析指标并预处理
selected_cols = [
    'Y染色体浓度', 
    '孕妇BMI', 
    '检测孕周数值', 
    '被过滤掉读段数的比例', 
    '在参考基因组上比对的比例',
    '年龄',
    'GC含量'
]
df_selected = df[selected_cols].copy()
df_selected.dropna(inplace=True)  # 删除缺失值

# 4. 计算相关系数
## Pearson相关（线性相关，假设近似正态分布）
pearson_corr = df_selected.corr(method='pearson')
print("=== Pearson 相关系数矩阵（含转换后孕周）===")
print(pearson_corr)

## Spearman相关（秩相关，不要求正态分布）
spearman_corr = df_selected.corr(method='spearman')
print("\n=== Spearman 相关系数矩阵（含转换后孕周）===")
print(spearman_corr)

# 5. 可视化相关性
## 5.1 Pearson相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson 相关性热图')
plt.tight_layout()
plt.savefig('Pearson.png',dpi=400)
plt.show()

## 5.2 Spearman相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman 相关性热图')
plt.tight_layout()
plt.savefig('Spearman.png',dpi=400)
plt.show()

## 5.3 Pairwise散点图+回归拟合
sns.pairplot(df_selected, kind='reg', diag_kind='kde')
plt.suptitle('变量间 Pairwise 关系与线性回归拟合', y=1.02)
plt.tight_layout()
plt.savefig('pairplot_gestation_converted.png',dpi=400)
plt.show()

## 5.4 单独绘制“Y染色体浓度与各指标”的回归图（带相关系数和p值）
plt.figure(figsize=(14, 10))
for i, var in enumerate(df_selected.columns[1:]):
    plt.subplot(2, 3, i+1)
    sns.regplot(x='Y染色体浓度', y=var, data=df_selected)
    plt.title(f'{var} vs Y染色体浓度')
    # 计算Pearson相关系数与p值
    r, p = stats.pearsonr(df_selected[var], df_selected['Y染色体浓度'])
    plt.text(
        0.1, 0.9, 
        f'Pearson r = {r:.3f}\np = {p:.3e}', 
        transform=plt.gca().transAxes, 
        bbox=dict(facecolor='white', alpha=0.8)
    )
plt.tight_layout()
plt.savefig('Y-Individual.png',dpi=400)
plt.show()