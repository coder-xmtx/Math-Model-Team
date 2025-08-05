import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpStatus

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ----------------------
# 1. 参数设置
# ----------------------
WEEKS = 24  # 计划周期
CAPACITY = 28200  # 每周产能（立方米）
SAFETY_STOCK_WEEKS = 2  # 安全库存周数
SAFETY_STOCK = CAPACITY * SAFETY_STOCK_WEEKS  # 安全库存量

# 原材料转换系数 (每立方米原材料可生产的产品立方米数)
CONVERSION_FACTORS = {
    'A': 1 / 0.6,   # A类原材料转换系数
    'B': 1 / 0.66,  # B类原材料转换系数
    'C': 1 / 0.72   # C类原材料转换系数
}

# ----------------------
# 2. 数据读取与处理
# ----------------------
def load_supplier_data():
    """读取供应商数据并进行预处理"""
    try:
        # 读取Top50供应商的供货量数据
        supply_data = pd.read_csv("data/Top50供应商供货量.csv")
        # 读取向Top50供应商的订货量数据
        order_data = pd.read_csv("data/企业向Top50供应商订货量.csv")
        
        # 提取供应商ID和材料类型
        suppliers = supply_data[['supplier_ID', 'material']]
        supplier_ids = suppliers['supplier_ID'].unique()
        num_suppliers = len(supplier_ids)
        
        # 提取每周数据（假设列名为W001到W240）
        weeks_columns = [f'W{i:03d}' for i in range(1, 241)]
        supply_weeks = supply_data[weeks_columns]
        order_weeks = order_data[weeks_columns]
        
        # 将240周数据按24周为一个周期划分，得到10个周期
        periods = []
        for i in range(10):  # 10个周期
            start_idx = i * WEEKS
            end_idx = start_idx + WEEKS
            period_data = supply_weeks.iloc[:, start_idx:end_idx].copy()
            period_data.columns = [f'W{i:03d}' for i in range(1, WEEKS+1)]
            periods.append(period_data)
        
        return suppliers, periods, supplier_ids, num_suppliers
        
    except Exception as e:
        print(f"数据读取错误: {e}")
        # 生成模拟数据用于测试
        print("生成模拟数据用于测试...")
        return generate_sample_data()

def generate_sample_data():
    """生成模拟数据用于测试"""
    num_suppliers = 50
    # 随机生成供应商ID和材料类型
    supplier_ids = [f'S{i:03d}' for i in range(1, num_suppliers+1)]
    materials = np.random.choice(['A', 'B', 'C'], size=num_suppliers)
    suppliers = pd.DataFrame({
        'supplier_ID': supplier_ids,
        'material': materials
    })
    
    # 生成10个周期的模拟供货数据
    periods = []
    for _ in range(10):
        period_data = pd.DataFrame()
        for week in range(1, WEEKS+1):
            # 为每个供应商生成每周的供货量（1000-5000立方米之间）
            period_data[f'W{week:03d}'] = np.random.randint(1000, 5000, size=num_suppliers)
        periods.append(period_data)
    
    return suppliers, periods, supplier_ids, num_suppliers

# 加载数据
suppliers, periods, supplier_ids, num_suppliers = load_supplier_data()

# 计算每个供应商在未来24周中每周的预期供货量S_{i,j}
# S_{i,j} = 0.9 * 10个周期中第j周的最大供货量
expected_supply = pd.DataFrame(index=supplier_ids)
for week in range(1, WEEKS+1):
    week_col = f'W{week:03d}'
    # 收集10个周期中该周的供货量
    week_data = [period[week_col].values for period in periods]
    # 计算每个供应商在该周的最大供货量
    max_week_supply = np.max(week_data, axis=0)
    # 乘以0.9作为预期供货量
    expected_supply[week_col] = max_week_supply * 0.9

# ----------------------
# 3. 建立与求解线性规划模型
# ----------------------
def solve_min_suppliers():
    """求解最小供应商数量问题"""
    # 创建问题实例，目标是最小化
    model = LpProblem("Minimize_Suppliers", LpMinimize)
    
    # 创建决策变量: x_{i} 表示是否选择供应商i（简化模型：一旦选择就在整个周期内供货）
    x = LpVariable.dicts("Supplier", supplier_ids, 0, 1, LpBinary)
    
    # 目标函数: 最小化供应商数量
    model += lpSum([x[i] for i in supplier_ids]), "Total_Suppliers"
    
    # 初始库存
    initial_stock = SAFETY_STOCK
    
    # 添加每周的约束条件，考虑库存变化
    current_stock = initial_stock
    for week in range(1, WEEKS+1):
        week_col = f'W{week:03d}'
        
        # 计算该周的总有效供应量（考虑原材料类型转换）
        total_supply = 0
        for i in supplier_ids:
            # 获取供应商i的材料类型
            material = suppliers[suppliers['supplier_ID'] == i]['material'].values[0]
            # 获取转换系数
            conversion = CONVERSION_FACTORS[material]
            # 获取预期供货量
            supply = expected_supply.loc[i, week_col]
            # 累加有效供应量
            total_supply += x[i] * supply * conversion
        
        # 约束条件: 总有效供应量 + 当前库存 >= 本周产能需求 + 安全库存
        # 同时更新下周库存
        constraint = total_supply + current_stock >= CAPACITY + SAFETY_STOCK
        model += constraint, f"Week_{week}_Supply_Constraint"
        
        # 更新当前库存（用于下周计算）
        current_stock = total_supply + current_stock - CAPACITY
    
    # 求解模型
    status = model.solve()
    
    # 输出结果
    print(f"求解状态: {status}, {LpStatus[status]}")
    print(f"最小供应商数量: {int(model.objective.value())}")
    
    # 收集选中的供应商
    selected_suppliers = [i for i in supplier_ids if x[i].varValue == 1]
    print(f"选中的供应商数量: {len(selected_suppliers)}")
    
    return selected_suppliers, model

# 求解问题
selected_suppliers, model = solve_min_suppliers()

# ----------------------
# 4. 结果分析与可视化
# ----------------------
def visualize_results(selected_suppliers):
    """可视化结果"""
    # 1. 供应商类型分布
    material_counts = {}
    for i in selected_suppliers:
        material = suppliers[suppliers['supplier_ID'] == i]['material'].values[0]
        material_counts[material] = material_counts.get(material, 0) + 1
    
    plt.figure(figsize=(12, 6))
    
    # 子图1: 选中供应商的材料类型分布
    plt.subplot(1, 2, 1)
    materials = list(material_counts.keys())
    counts = list(material_counts.values())
    plt.bar(materials, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('选中供应商的材料类型分布')
    plt.xlabel('材料类型')
    plt.ylabel('供应商数量')
    plt.xticks(materials)
    
    # 2. 每周供应量分析
    plt.subplot(1, 2, 2)
    weekly_supply = []
    weekly_stock = []
    current_stock = SAFETY_STOCK
    
    for week in range(1, WEEKS+1):
        week_col = f'W{week:03d}'
        total = 0
        for i in selected_suppliers:
            material = suppliers[suppliers['supplier_ID'] == i]['material'].values[0]
            conversion = CONVERSION_FACTORS[material]
            supply = expected_supply.loc[i, week_col]
            total += supply * conversion
        weekly_supply.append(total)
        
        # 计算库存变化
        current_stock = total + current_stock - CAPACITY
        weekly_stock.append(current_stock)
    
    x = range(1, WEEKS+1)
    plt.plot(x, weekly_supply, 'o-', color='#d62728', label='总有效供应量')
    plt.plot(x, weekly_stock, 's-', color='#2ca02c', label='周末库存量')
    plt.axhline(y=CAPACITY, color='#9467bd', linestyle='--', label='产能需求')
    plt.axhline(y=SAFETY_STOCK, color='#ff7f0e', linestyle='-.', label='安全库存量')
    plt.title('每周有效供应量与库存变化')
    plt.xlabel('周数')
    plt.ylabel('数量 (产品立方米)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('供应商选择结果分析.png', dpi=300)
    plt.show()
    
    # 3. 各类原材料的总供应量
    material_supply = {'A': 0, 'B': 0, 'C': 0}
    for i in selected_suppliers:
        material = suppliers[suppliers['supplier_ID'] == i]['material'].values[0]
        conversion = CONVERSION_FACTORS[material]
        total = 0
        for week in range(1, WEEKS+1):
            week_col = f'W{week:03d}'
            total += expected_supply.loc[i, week_col]
        material_supply[material] += total * conversion / WEEKS  # 平均每周
    
    plt.figure(figsize=(8, 6))
    plt.pie(material_supply.values(), labels=material_supply.keys(), 
            autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('各类原材料的平均每周有效供应占比')
    plt.savefig('原材料供应占比.png', dpi=300)
    plt.show()

# 可视化结果
if selected_suppliers:
    visualize_results(selected_suppliers)

# ----------------------
# 5. 生成详细结果报告
# ----------------------
def generate_report(selected_suppliers):
    """生成详细的结果报告"""
    with open('供应商选择结果报告.txt', 'w', encoding='utf-8') as f:
        f.write("===== 最小供应商数量分析报告 =====\n\n")
        f.write(f"计划周期: {WEEKS}周\n")
        f.write(f"每周产能需求: {CAPACITY}立方米\n")
        f.write(f"安全库存量: {SAFETY_STOCK}立方米 (相当于{SAFETY_STOCK_WEEKS}周产能)\n\n")
        
        f.write(f"最小供应商数量: {len(selected_suppliers)}\n\n")
        f.write("选中的供应商列表:\n")
        for i, supplier in enumerate(selected_suppliers, 1):
            material = suppliers[suppliers['supplier_ID'] == supplier]['material'].values[0]
            f.write(f"{i}. 供应商ID: {supplier}, 材料类型: {material}\n")
        
        f.write("\n各类原材料供应商数量:\n")
        material_counts = {}
        for i in selected_suppliers:
            material = suppliers[suppliers['supplier_ID'] == i]['material'].values[0]
            material_counts[material] = material_counts.get(material, 0) + 1
        for material, count in material_counts.items():
            f.write(f"- {material}类原材料: {count}家供应商\n")

# 生成报告
if selected_suppliers:
    generate_report(selected_suppliers)
