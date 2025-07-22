import numpy as np

np.random.seed(42)  # 确保结果可重复

# 定义盆地数据（根据表格输入）
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

# 预处理孔隙度：计算均值和标准差（假设范围覆盖均值±2σ）
for basin in basins:
    porosity_str = basin['porosity']
    if '~' in porosity_str:
        phi_min, phi_max = map(float, porosity_str.split('~'))
        basin['phi_mean'] = (phi_min + phi_max) / 2
        basin['phi_std'] = (phi_max - phi_min) / 4  # 标准差 = 范围/4
    else:
        phi_val = float(porosity_str)
        basin['phi_mean'] = phi_val
        basin['phi_std'] = 0.0  # 固定值，标准差为0

# 密度参数（假设均值为700，标准差= (770-630)/4=35）
rho_mean = 700
rho_std = 35

# 蒙特卡洛模拟
num_simulations = 100000
total_co2 = []

for _ in range(num_simulations):
    total_mt = 0.0
    for basin in basins:
        # 单位转换
        A = basin['area'] * 1e10  # 转换为平方米
        h = basin['h']
        
        # 生成正态分布随机值（孔隙度）
        phi = np.random.normal(basin['phi_mean'], basin['phi_std'])
        phi = np.clip(phi, basin['phi_mean'] - 2*basin['phi_std'], basin['phi_mean'] + 2*basin['phi_std'])  # 截断
        
        # 生成正态分布随机值（密度）
        rho = np.random.normal(rho_mean, rho_std)
        rho = np.clip(rho, 630, 770)  # 截断
        
        # 计算封存量
        M_kg = 0.01 * A * h * phi * rho * 0.03  # S_eff=3%
        total_mt += M_kg / 1e9  # 转换为百万吨
        
    total_co2.append(total_mt)

# 计算结果统计
mean = np.mean(total_co2)
std = np.std(total_co2)
confidence_95 = np.percentile(total_co2, [2.5, 97.5])

# 输出结果
print(f"蒙特卡洛模拟（正态分布，{num_simulations}次）结果：")
print(f"CO₂总封存量均值：{mean:.2f} Mt")
print(f"标准差：{std:.2f} Mt")
print(f"95%置信区间：[{confidence_95[0]:.2f}, {confidence_95[1]:.2f}] Mt")