import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 孕周字符串转数值
def convert_gestational_week(week_str):
    if pd.isna(week_str):
        return None
    match = re.match(r'(\d+)w(?:\+(\d+))?', str(week_str))
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return weeks + days / 7.0
    return None

# 数据降噪处理 - 滑动平均滤波
def moving_average_denoise(data, window_size=4):
    """
    滑动平均滤波进行数据降噪
    data: 输入数据
    window_size: 滑动窗口大小，需为正整数，3-7
    返回降噪后的数据
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保窗口大小为奇数
    pad_width = window_size // 2
    # 边缘处理：首尾填充相同值
    padded_data = np.pad(data, pad_width, mode='edge')
    # 滑动平均计算
    denoised_data = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
    return denoised_data

# 数据无量纲化处理 - Min-Max标准化
def normalize_data(data):
    """
    将数据归一化到[0, 1]区间
    返回归一化后的数据、最小值和最大值
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:  # 避免除以零
        return np.zeros_like(data), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val

def denormalize_data(normalized_data, min_val, max_val):
    """将归一化的数据还原为原始尺度"""
    if max_val == min_val:
        return np.full_like(normalized_data, min_val)
    return normalized_data * (max_val - min_val) + min_val

# 1. 读取Excel并预处理
df = pd.read_excel('./data/data-total.xlsx')
df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)

# 选择母序列（Y染色体浓度）和子序列（4个影响因素）
mother_col = 'Y染色体浓度'
child_cols = ['孕妇BMI', '年龄', '检测孕周数值', '在参考基因组上比对的比例']
selected_cols = [mother_col] + child_cols
df_gm = df[selected_cols].copy().dropna()  # 删除缺失值

# 2. 数据降噪处理
window_size = 6

# 对母序列进行降噪
X0_raw = df_gm[mother_col].values  # 原始母序列（未降噪）
X0_denoised = moving_average_denoise(X0_raw, window_size)  # 降噪后的母序列

# 对子序列进行降噪
X_raw = df_gm[child_cols].values  # 原始子序列（未降噪）
X_denoised = np.zeros_like(X_raw)  # 初始化降噪后的子序列

for i in range(X_raw.shape[1]):
    X_denoised[:, i] = moving_average_denoise(X_raw[:, i], window_size)

# 3. 数据无量纲化处理（基于降噪后的数据）
# 对母序列进行归一化
X0_normalized, X0_min, X0_max = normalize_data(X0_denoised)

# 对子序列进行归一化
X_normalized = np.zeros_like(X_denoised)
X_mins = np.zeros(X_denoised.shape[1])
X_maxs = np.zeros(X_denoised.shape[1])

for i in range(X_denoised.shape[1]):
    X_normalized[:, i], X_mins[i], X_maxs[i] = normalize_data(X_denoised[:, i])

# 使用归一化后的数据进行建模
X0 = X0_normalized  # 归一化后的母序列
X = X_normalized    # 归一化后的子序列
n = len(X0)         # 样本量

def ago(x):
    """一次累加生成（1-AGO）：cumsum"""
    return np.cumsum(x)

X0_ago = ago(X0)  # 母序列累加
X_agos = np.array([ago(X[:, i]) for i in range(X.shape[1])]).T  # 子序列各自累加后转置

# 构造B矩阵（系数矩阵）和Y矩阵（目标向量）
B = np.zeros((n-1, len(child_cols) + 1))
Y = np.zeros((n-1, 1))

for k in range(1, n):
    # 母序列背景值 Z0
    Z0 = 0.5 * X0_ago[k] + 0.5 * X0_ago[k-1]
    Y[k-1, 0] = X0[k]  # 母序列原始值（k≥1）
    B[k-1, 0] = -Z0    # 发展系数a的系数列
    
    # 子序列背景值 Zi
    for i in range(len(child_cols)):
        Zi = 0.5 * X_agos[k, i] + 0.5 * X_agos[k-1, i]
        B[k-1, i+1] = Zi  # 驱动系数bi的系数列

# 最小二乘求解参数：params = [a, b1, b2, b3, b4]^T
params = np.linalg.inv(B.T @ B) @ B.T @ Y
a = params[0, 0]       # 发展系数
b = params[1:, 0]      # 各子序列的驱动系数

# 模型拟合：对已有数据回代
X0_fit_normalized = np.zeros_like(X0)
X0_fit_normalized[0] = X0[0]  # 第一个值与原始值一致

for k in range(1, n):
    Z0 = 0.5 * X0_ago[k] + 0.5 * X0_ago[k-1]
    Zi_list = [0.5 * X_agos[k, i] + 0.5 * X_agos[k-1, i] for i in range(len(child_cols))]
    X0_fit_normalized[k] = -a * Z0 + np.sum(b * np.array(Zi_list))

# 将拟合结果反归一化到原始尺度
X0_fit = denormalize_data(X0_fit_normalized, X0_min, X0_max)

# 残差与相对误差（使用原始尺度数据计算）
residuals = X0_raw - X0_fit  # 与原始未降噪数据比较
relative_errors = np.abs(residuals) / X0_raw * 100  # 相对误差（%）

# 后验差检验（评估模型精度）
S0 = np.std(X0_raw, ddof=1)        # 母序列原始标准差
S_res = np.std(residuals, ddof=1)  # 残差标准差
C = S_res / S0                 # 后验差比（越小精度越高）
P = np.sum(np.abs(residuals) < 0.6745 * S0) / n  # 小误差概率（越大精度越高）

# 模型精度等级判断
def gm_accuracy(C, P):
    if C < 0.35 and P > 0.95:
        return "好"
    elif C < 0.5 and P > 0.8:
        return "合格"
    elif C < 0.65 and P > 0.7:
        return "基本合格"
    else:
        return "不合格"

accuracy = gm_accuracy(C, P)

# 打印模型参数与精度
print("=== GM(1,N)模型参数 ===")
print(f"发展系数 a = {a:.6f}")
for i, col in enumerate(child_cols):
    print(f"驱动系数 b_{i+1}（{col}）= {b[i]:.6f}")

print("\n=== 模型精度检验 ===")
print(f"平均相对误差：{np.mean(relative_errors):.2f}%")
print(f"后验差比 C = {C:.6f}")
print(f"小误差概率 P = {P:.6f}")
print(f"模型精度等级：{accuracy}")

# 可视化1：原始数据与降噪后数据对比
plt.figure(figsize=(10, 6))
plt.plot(X0_raw, 'o-', label='原始Y染色体浓度', alpha=0.6)
plt.plot(X0_denoised, 's-', label=f'降噪后Y染色体浓度 (窗口大小={window_size})', color='orange')
plt.xlabel('样本索引')
plt.ylabel('Y染色体浓度')
plt.title('原始数据与降噪后数据对比')
plt.legend()
plt.grid(True)
plt.savefig('origin-quzao.png', dpi=400)
plt.show()

# 可视化2：实际值 vs 拟合值
plt.figure(figsize=(10, 6))
plt.plot(X0_raw, 'o-', label='实际Y染色体浓度', alpha=0.6)
plt.plot(X0_fit, 's-', label='GM(1,N)拟合值', color='green')
plt.xlabel('样本索引')
plt.ylabel('Y染色体浓度')
plt.title('GM(1,N)模型拟合结果（含降噪处理）')
plt.legend()
plt.grid(True)
plt.savefig('gray_fitting_quzao.png', dpi=400)
plt.show()

# 输出模型表达式
model_expr = f"X⁽⁰⁾(k) = {-a:.6f} \cdot Z⁽¹⁾₀(k) "
for i, col in enumerate(child_cols):
    model_expr += f"+ {b[i]:.6f} \cdot Z⁽¹⁾_{i+1}(k) "
print("\n=== GM(1,N)模型表达式 ===")
print(model_expr)
    