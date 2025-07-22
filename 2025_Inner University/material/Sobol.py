import numpy as np
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol

# 定义盆地数据
basins = [
    # 陆上盆地
    {'name': '塔里木盆地', 'area': 53, 'h': 300, 'porosity': '0.1~0.234'},
    {'name': '鄂尔多斯盆地', 'area': 37, 'h': 300, 'porosity': '0.12~0.2'},
    {'name': '渤海湾盆地（陆上）', 'area': 13.3, 'h': 200, 'porosity': '0.1~0.2'},
    {'name': '松辽盆地', 'area': 26, 'h': 200, 'porosity': '0.15~0.2'},
    {'name': '准噶尔盆地', 'area': 38, 'h': 300, 'porosity': '0.15~0.169'},
    {'name': '河淮盆地', 'area': 27, 'h': 300, 'porosity': '0.2'},
    {'name': '苏北盆地', 'area': 18, 'h': 300, 'porosity': '0.2'},
    {'name': '二连盆地', 'area': 10, 'h': 200, 'porosity': '0.15'},
    {'name': '四川盆地', 'area': 26, 'h': 300, 'porosity': '0.05'},
    {'name': '吐鲁番盆地', 'area': 5.35, 'h': 300, 'porosity': '0.15'},
    {'name': '哈密盆地', 'area': 5.35, 'h': 300, 'porosity': '0.15'},
    {'name': '江汉盆地', 'area': 1.9, 'h': 150, 'porosity': '0.2'},
    {'name': '洞庭盆地', 'area': 1.9, 'h': 150, 'porosity': '0.2'},
    {'name': '三江盆地', 'area': 3.373, 'h': 200, 'porosity': '0.15'},
    {'name': '沁水盆地', 'area': 2.3923, 'h': 300, 'porosity': '0.15'},
    {'name': '柴达木盆地', 'area': 25.7768, 'h': 50, 'porosity': '0.15'},
    {'name': '海拉尔盆地', 'area': 7.048, 'h': 100, 'porosity': '0.15'},
    {'name': '南襄盆地', 'area': 4.6, 'h': 100, 'porosity': '0.15'},
    # 海上盆地
    {'name': '东海盆地', 'area': 46, 'h': 300, 'porosity': '0.15'},
    {'name': '南黄海盆地', 'area': 8.5, 'h': 300, 'porosity': '0.15'},
    {'name': '渤海湾盆地（海上）', 'area': 6.7, 'h': 300, 'porosity': '0.2'},
    {'name': '珠江口盆地', 'area': 15, 'h': 200, 'porosity': '0.15'},
    {'name': '莺歌海盆地', 'area': 6, 'h': 300, 'porosity': '0.15'},
    {'name': '北黄海盆地', 'area': 7.13, 'h': 300, 'porosity': '0.2'},
    {'name': '北部湾盆地', 'area': 3.98, 'h': 300, 'porosity': '0.15'},
]

# 预处理孔隙度参数
for basin in basins:
    porosity_str = basin['porosity']
    if '~' in porosity_str:
        phi_min, phi_max = map(float, porosity_str.split('~'))
        basin['phi_mean'] = (phi_min + phi_max) / 2
        basin['phi_std'] = (phi_max - phi_min) / 4
    else:
        phi_val = float(porosity_str)
        basin['phi_mean'] = phi_val
        basin['phi_std'] = 0.0

# 定义Sobol分析问题
problem = {
    'num_vars': 1,
    'names': ['rho'],
    'bounds': [[630, 770]]  # 密度范围
}

# 生成Sobol样本（自动满足样本数要求）
n_samples = 5000
param_values = saltelli.sample(problem, n_samples)

# 模型计算函数
def model(param_values):
    Y = np.zeros(param_values.shape[0])
    for i, rho in enumerate(param_values):
        total_mt = 0.0
        for basin in basins:
            A = basin['area'] * 1e10
            h = basin['h']
            phi = np.random.normal(basin['phi_mean'], basin['phi_std'])
            phi = np.clip(phi, basin['phi_mean'] - 2*basin['phi_std'], 
                          basin['phi_mean'] + 2*basin['phi_std'])
            # 提取 rho 的标量值（rho[0]）
            M_kg = 0.01 * A * h * phi * rho[0] * 0.03
            total_mt += M_kg / 1e9
        Y[i] = total_mt
    return Y

# 运行模型
Y = model(param_values)

print(param_values.shape)
print(Y.shape)

# 执行Sobol分析
Si = sobol.analyze(problem, Y, print_to_console=False)

# 输出灵敏度结果
print(f"一阶灵敏度指数 (S1): {Si['S1'][0]:.4f}")
print(f"总效应指数 (ST): {Si['ST'][0]:.4f}")