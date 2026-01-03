import pandas as pd
import matplotlib.pyplot as plt

# 加载光谱数据
spectrum_data = pd.read_csv('data/Problem_1.csv')

# 加载 melanopicPhotopic 数据
melanopic_data = pd.read_csv('data/melanopicPhotopic.csv')

# 定义函数计算 MDER
def calculate_MDER(spectrum, melanopic):
    # 计算黑视素通量
    melanopic_flux = 832 * sum(spectrum['SPD'] * melanopic['ipRGC'])
    # 计算光通量
    photopic_flux = 683 * sum(spectrum['SPD'] * melanopic['bright-vision-curve'])
    # 计算 MPR
    MPR = melanopic_flux / photopic_flux
    # 计算 MEDR
    MDER = MPR / 832 * 1000 / 1.326
    return MDER

# 计算 MDER
MDER_result = calculate_MDER(spectrum_data, melanopic_data)

print(f'MDER 的值为: {MDER_result}')

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制黑视素敏感曲线
ax.plot(melanopic_data['wavelength'], melanopic_data['ipRGC'], 
        label='黑视素敏感曲线 ($S_{mel}(λ)$)', color='blue', linewidth=2)

# 绘制明视觉光谱曲线
ax.plot(melanopic_data['wavelength'], melanopic_data['bright-vision-curve'], 
        label='明视觉光谱曲线 ($V(λ)$)', color='red', linewidth=2, linestyle='--')

# 设置坐标轴标签和标题
ax.set_xlabel('波长 (nm)', fontsize=12)
ax.set_ylabel('相对灵敏度', fontsize=12)
ax.set_title('黑视素敏感曲线与明视觉光谱曲线对比', fontsize=14, pad=20)

# 设置坐标轴范围
ax.set_xlim(380, 780)
ax.set_ylim(0, max(max(melanopic_data['ipRGC']), max(melanopic_data['bright-vision-curve'])) * 1.1)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()