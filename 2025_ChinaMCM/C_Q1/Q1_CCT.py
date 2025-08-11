import numpy as np
from scipy.optimize import brentq

# 给定的色品坐标 (x, y)
x = 0.3840445003795555
y = 0.3767800565662821

# --- 黑体轨迹的 Chebyshev 法 ---
# 转换为 CIE 1960 (u, v)
denominator = -2 * x + 12 * y + 3
u_c = 4 * x / denominator
v_c = 6 * y / denominator

# 定义黑体轨迹函数
def u_bar(T):
    numerator = 0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T**2
    denominator = 1 + 8.42420235e-4 * T + 7.08145163e-7 * T**2
    return numerator / denominator

def v_bar(T):
    numerator = 0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T**2
    denominator = 1 - 2.89741816e-5 * T + 1.61456053e-7 * T**2
    return numerator / denominator

# 数值微分 (中心差分)
def derivative(func, T, h=0.1):
    return (func(T + h) - func(T - h)) / (2 * h)

# 目标函数
def objective(T):
    u_t = u_bar(T)
    v_t = v_bar(T)
    du_dT = derivative(u_bar, T)
    dv_dT = derivative(v_bar, T)
    return du_dT * (u_t - u_c) + dv_dT * (v_t - v_c)

# 求解相关色温 (使用布伦特法)
try:
    T_c = brentq(objective, 1000, 15000, xtol=1e-5)
    print("==黑体轨迹的 Chebyshev 法==")
    print(f"相关色温 T_c = {T_c:.4f} K")
except ValueError:
    # 如果端点符号相同，尝试分段搜索
    T_values = np.linspace(1000, 15000, 100)
    f_values = [objective(T) for T in T_values]
    for i in range(len(T_values) - 1):
        if f_values[i] * f_values[i+1] <= 0:
            T_c = brentq(objective, T_values[i], T_values[i+1], xtol=1e-5)
            print("==黑体轨迹的 Chebyshev 法==")
            print(f"相关色温 T_c = {T_c:.4f} K")
            break


# --- McCamy近似公式法 ---
# 计算 n
n = (x - 0.3320) / (y - 0.1858)

# 计算相关色温 T
T = -437 * n**3 + 3601 * n**2 - 6861 * n + 5514.31

print("==McCamy 近似公式法==")
print(f"相关色温 T= {T:.4f} K")