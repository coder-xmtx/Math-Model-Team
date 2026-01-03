import pandas as pd
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 正常显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 正常显示符号
plt.rcParams['axes.unicode_minus'] = False

# 加载 CIE 标准观察者函数数据
cie_data = pd.read_csv('data/CIE_xyz_1931_2deg.csv')

# 加载光谱数据
spd_data = pd.read_csv('data/Problem_1.csv')

# 合并数据
merged_data = pd.merge(cie_data, spd_data, on='wavelength')

# 定义波长间隔
delta_lambda = 1

# 计算归一化系数 k
delta_lambda = 1
k = 100 / ((spd_data['SPD'] * cie_data['y'] * delta_lambda).sum())

# 计算 X、Y 和 Z
X = k*((merged_data['SPD'] * merged_data['x']).sum() * delta_lambda)
Y = k*((merged_data['SPD'] * merged_data['y']).sum() * delta_lambda)
Z = k*((merged_data['SPD'] * merged_data['z']).sum() * delta_lambda)

# 计算色品坐标
x = X / (X + Y + Z)
y = Y / (X + Y + Z)

# 打印色品坐标
print(f'色品坐标 (x, y): ({x}, {y})')

# 绘制 CIE 标准观察者函数数据
plt.figure()
plt.plot(cie_data['wavelength'], cie_data[['x', 'y', 'z']])
plt.xlabel('波长 (nm)')
plt.ylabel('刺激值')
plt.title('CIE 标准观察者函数')
plt.legend(['x', 'y', 'z'])
plt.savefig('CIE_standard_observer_functions.png')

# 绘制光谱数据
plt.figure()
plt.plot(spd_data['wavelength'], spd_data['SPD'])
plt.xlabel('波长 (nm)')
plt.ylabel('SPD')
plt.title('光谱数据')
plt.savefig('spectral_data.png')