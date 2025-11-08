import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取Excel文件
df = pd.read_excel("data/data-total.xlsx")

# 2. 提取列数据：Y染色体浓度、在参考基因组上比对的比例
y_chrom_conc = df["Y染色体浓度"]  # Y染色体浓度
maternal_bmi = df["在参考基因组上比对的比例"]  # 在参考基因组上比对的比例

# 3. 线性回归拟合
slope, intercept, r_value, p_value, std_err = stats.linregress(y_chrom_conc, maternal_bmi)

# 4. 绘制散点图 + 拟合直线
plt.figure(figsize=(10, 6))  # 设置图大小

# 散点图
plt.scatter(y_chrom_conc, maternal_bmi, color="skyblue", alpha=0.7, label="原始数据")

# 拟合直线
fit_line = slope * y_chrom_conc + intercept
plt.plot(y_chrom_conc, fit_line, color="red", linewidth=2, 
         label=f"线性拟合: y={slope:.4f}x + {intercept:.4f}\n$R^2$={r_value**2:.4f}, p={p_value:.4f}")

# 图表美化
plt.xlabel("Y染色体浓度", fontsize=12)
plt.ylabel("在参考基因组上比对的比例", fontsize=12)
plt.title("Y染色体浓度与在参考基因组上比对的比例的关系及线性拟合", fontsize=14)
plt.legend(fontsize=10)
plt.grid(linestyle="--", alpha=0.5)

# 5. 显示图表
plt.savefig('Y-gene.png',dpi=400)
plt.show()

# 6. 打印拟合统计结果
print("=== 线性拟合结果 ===")
print(f"斜率 (slope): {slope:.4f}")
print(f"截距 (intercept): {intercept:.4f}")
print(f"相关系数 (r): {r_value:.4f}")
print(f"决定系数 ($R^2$): {r_value**2:.4f}")
print(f"p值 (p-value): {p_value:.4f}")
print(f"标准误差 (std_err): {std_err:.4f}")