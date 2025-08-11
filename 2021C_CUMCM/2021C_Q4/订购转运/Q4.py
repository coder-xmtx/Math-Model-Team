import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpVariable, PULP_CBC_CMD, LpMinimize, LpInteger, lpSum, value, LpStatus
import os
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 数据读取与预处理 ----------------------
def load_data():
    """读取并预处理所有需要的数据"""
    print("正在读取数据...")
    
    # 读取Top50供应商数据
    top50_supply = pd.read_csv("data/Top50供应商供货量.csv")
    top50_order = pd.read_csv("data/企业向Top50供应商订货量.csv")
    
    # 读取转运商损耗率数据
    transporter_loss = pd.read_csv("data/转运商运输损耗率.csv")
    
    # 数据预处理
    # 提取周数据列
    week_cols = [col for col in top50_supply.columns if col.startswith('W')]
    
    # 转换供应商数据格式，使其更易于处理
    def reshape_supplier_data(df):
        # 融化数据框，将周列转换为行
        melted = df.melt(id_vars=['supplier_ID', 'material'], 
                         value_vars=week_cols,
                         var_name='week', 
                         value_name='quantity')
        # 提取周数数字
        melted['week_num'] = melted['week'].apply(lambda x: int(x[1:]))
        return melted.sort_values(['supplier_ID', 'week_num'])
    
    # 转换转运商数据格式
    def reshape_transporter_data(df):
        melted = df.melt(id_vars=['transporter_ID'], 
                         value_vars=[col for col in df.columns if col.startswith('W')],
                         var_name='week', 
                         value_name='loss_rate')
        melted['week_num'] = melted['week'].apply(lambda x: int(x[1:]))
        # 转换损耗率为小数
        melted['loss_rate'] = melted['loss_rate'] / 100
        return melted.sort_values(['transporter_ID', 'week_num'])
    
    # 处理供应商数据
    top50_supply_melted = reshape_supplier_data(top50_supply)
    top50_order_melted = reshape_supplier_data(top50_order)
    
    # 处理转运商数据
    transporter_loss_melted = reshape_transporter_data(transporter_loss)
    
    print("数据读取完成")
    return {
        'top50_supply': top50_supply,
        'top50_order': top50_order,
        'top50_supply_melted': top50_supply_melted,
        'top50_order_melted': top50_order_melted,
        'transporter_loss': transporter_loss,
        'transporter_loss_melted': transporter_loss_melted,
        'week_cols': week_cols
    }

# ---------------------- 计算供应商供货能力 ----------------------
def calculate_supplier_capacity(data):
    """计算每个供应商每周的供货能力 S_{i,j}"""
    print("正在计算供应商供货能力...")
    
    top50_supply = data['top50_supply']
    week_cols = data['week_cols']
    suppliers = top50_supply['supplier_ID'].unique()
    materials = top50_supply['material'].unique()
    
    # 240周数据按24周为一个周期，分成10个周期
    periods = 10
    period_weeks = 24
    
    # 创建一个字典存储每个供应商每周的供货能力
    supplier_capacity = {}
    
    for supplier_id in suppliers:
        supplier_data = top50_supply[top50_supply['supplier_ID'] == supplier_id]
        material = supplier_data['material'].iloc[0]
        
        for j in range(1, period_weeks + 1):  # 未来24周
            # 提取每个周期的第j周数据
            week_values = []
            for k in range(periods):
                week_idx = k * period_weeks + j - 1  # 转换为索引
                if week_idx < len(week_cols):
                    week_col = week_cols[week_idx]
                    week_values.append(supplier_data[week_col].iloc[0])
            
            # 计算S_{i,j} = 0.9 * max(s_{i,j,k})
            max_val = max(week_values) if week_values else 0
            s_ij = 0.9 * max_val
            supplier_capacity[(supplier_id, j)] = {
                'capacity': s_ij,
                'material': material
            }
    
    print("供应商供货能力计算完成")
    return supplier_capacity

# ---------------------- 订购方案优化 ----------------------
def optimize_order_plan(data, supplier_capacity):
    """优化订购方案"""
    print("正在优化订购方案...")
    
    # 模型参数
    P = 33000  # 每周产能（立方米）
    R0 = 2 * P  # 初始库存，满足两周生产需求
    G = {'A': 1.2, 'B': 1.1, 'C': 1.0}  # 购买价格系数
    H = 0.1  # 单位存储价格系数
    f = {'A': 1/0.6, 'B': 1/0.66, 'C': 1/0.72}  # 材料转换为产能的系数
    L = 0.98  # 产能转化损失系数
    weeks = 24  # 计划周期
    
    # 获取供应商和材料信息
    suppliers = data['top50_supply']['supplier_ID'].unique()
    supplier_materials = {s: data['top50_supply'][data['top50_supply']['supplier_ID'] == s]['material'].iloc[0] 
                         for s in suppliers}
    
    # 创建问题 - 多目标优化，这里采用加权法处理
    prob = LpProblem("Raw_Material_Order_Optimization", LpMinimize)
    
    # 决策变量：x_{i,j} 第j周是否从第i家供应商订货
    x = LpVariable.dicts("Order", 
                         [(i, j) for i in suppliers for j in range(1, weeks+1)],
                         0, 1, LpInteger)
    
    # 中间变量：每周各类材料的总订货量 - 修正键的定义方式
    V = LpVariable.dicts("TotalVolume", 
                         [(t, j) for t in ['A', 'B', 'C'] for j in range(1, weeks+1)],
                         0, None)
    
    # 中间变量：每周库存
    R = LpVariable.dicts("Inventory", 
                         [j for j in range(0, weeks+1)],
                         0, None)
    
    # 设置初始库存
    prob += R[0] == R0, "Initial_Inventory"
    
    # 计算每周各类材料的总订货量
    for t in ['A', 'B', 'C']:
        for j in range(1, weeks+1):
            # 确保只选择提供该类材料的供应商
            relevant_suppliers = [i for i in suppliers if supplier_materials[i] == t]
            # 只有当有相关供应商时才添加约束
            if relevant_suppliers:
                prob += V[(t, j)] == lpSum([x[(i, j)] * supplier_capacity[(i, j)]['capacity'] 
                                           for i in relevant_suppliers]), f"TotalVolume_{t}_{j}"
            else:
                # 如果没有该类材料的供应商，设置为0
                prob += V[(t, j)] == 0, f"TotalVolume_{t}_{j}_NoSuppliers"
    
    # 库存约束
    for j in range(1, weeks+1):
        # 总有效产能 = 各类材料有效产能之和
        total_effective = lpSum([V[(t, j)] * f[t] * L for t in ['A', 'B', 'C']])
        # 库存递推关系
        prob += R[j] == total_effective + R[j-1] - P, f"Inventory_Recurrence_{j}"
        # 库存约束：需满足本周产能和两周库存需求
        prob += total_effective + R[j-1] >= P + R0, f"Inventory_Constraint_{j}"
    
    # 目标函数：多目标优化，这里采用加权组合
    # 1. 最大化A类与C类的差值 (VA - VC)
    # 2. 最小化总成本
    
    # 计算总A类和总C类材料量
    total_A = lpSum([V[('A', j)] for j in range(1, weeks+1)])
    total_C = lpSum([V[('C', j)] for j in range(1, weeks+1)])
    
    # 计算总成本
    total_cost = lpSum([(G[t] + H) * V[(t, j)] for t in ['A', 'B', 'C'] for j in range(1, weeks+1)])
    
    # 组合目标函数（使用权重将最大化问题转为最小化问题）
    weight = 1e6  # 权重系数，根据实际情况调整
    prob += total_cost - weight * (total_A - total_C), "Combined_Objectives"
    
    # 求解问题
    prob.solve(PULP_CBC_CMD(msg=0,timeLimit=600))
    
    print(f"订购方案求解状态: {LpStatus[prob.status]}")
    
    # 提取结果
    order_plan = {}
    for i in suppliers:
        for j in range(1, weeks+1):
            z = value(x[(i, j)]) or 0.0
            if z > 0.5:  # 大于0.5视为订货
                order_plan[(i, j)] = {
                    'quantity': supplier_capacity[(i, j)]['capacity'],
                    'material': supplier_materials[i]
                }
    
    # 计算每周各类材料的订购量
    weekly_volume = {t: [0]*(weeks+1) for t in ['A', 'B', 'C']}  # 索引0不用
    for (i, j), details in order_plan.items():
        t = details['material']
        weekly_volume[t][j] += details['quantity']
    
    # 计算库存变化
    inventory = [0]*(weeks+1)
    inventory[0] = R0
    for j in range(1, weeks+1):
        total_effective = sum([weekly_volume[t][j] * f[t] * L for t in ['A', 'B', 'C']])
        inventory[j] = total_effective + inventory[j-1] - P
    
    return {
        'status': LpStatus[prob.status],
        'order_plan': order_plan,
        'weekly_volume': weekly_volume,
        'inventory': inventory,
        'total_A': value(total_A),
        'total_C': value(total_C),
        'total_cost': value(total_cost)
    }

# ---------------------- 转运方案优化 ----------------------
def optimize_transport_plan(data, order_plan):
    """优化转运方案"""
    print("正在优化转运方案...")
    
    # 获取数据
    transporter_loss_melted = data['transporter_loss_melted']
    weeks = 24  # 计划周期
    
    # 获取所有转运商
    transporters = transporter_loss_melted['transporter_ID'].unique()
    
    # 提取每个转运商每周的损耗率
    loss_rates = {}
    for idx, row in transporter_loss_melted.iterrows():
        # 只关注未来24周
        if 1 <= row['week_num'] <= weeks:
            loss_rates[(row['transporter_ID'], row['week_num'])] = row['loss_rate']
    
    # 获取所有有订单的供应商和周
    supplier_weeks = list(order_plan.keys())
    suppliers = list(set([i for i, j in supplier_weeks]))
    
    # 创建问题
    prob = LpProblem("Transportation_Optimization", LpMinimize)
    
    # 决策变量：x_{i,j,k} 第k周，供应商i的货物由转运商j运输的量
    x = LpVariable.dicts("Transport", 
                         [(i, j, k) for i, k in supplier_weeks for j in transporters],
                         0, None)
    
    # 目标函数：最小化总损耗
    prob += lpSum([x[(i, j, k)] * loss_rates[(j, k)] 
                  for i, k in supplier_weeks for j in transporters]), "Total_Loss"
    
    # 约束条件1：转运商每周转运量不超过6000立方米
    for j in transporters:
        for k in range(1, weeks+1):
            prob += lpSum([x[(i, j, k)] for i in suppliers if (i, k) in supplier_weeks]) <= 6000, \
                    f"Transporter_Capacity_{j}_{k}"
    
    # 约束条件2：转运量等于供货量
    for i, k in supplier_weeks:
        prob += lpSum([x[(i, j, k)] for j in transporters]) == order_plan[(i, k)]['quantity'], \
                f"Supply_Demand_{i}_{k}"
    
    # 求解问题
    prob.solve(PULP_CBC_CMD(msg=0,timeLimit=600))
    
    print(f"转运方案求解状态: {LpStatus[prob.status]}")
    
    # 提取结果
    transport_plan = {}
    for i, k in supplier_weeks:
        for j in transporters:
            qty = value(x[(i, j, k)])
            if qty > 1e-6:  # 忽略极小值
                if (i, k) not in transport_plan:
                    transport_plan[(i, k)] = []
                transport_plan[(i, k)].append({
                    'transporter': j,
                    'quantity': qty,
                    'loss_rate': loss_rates[(j, k)]
                })
    
    # 计算每周各转运商的转运量
    weekly_transport = {j: [0]*(weeks+1) for j in transporters}  # 索引0不用
    for (i, k), details in transport_plan.items():
        for d in details:
            j = d['transporter']
            weekly_transport[j][k] += d['quantity']
    
    # 计算每周总损耗
    weekly_loss = [0]*(weeks+1)  # 索引0不用
    for (i, k), details in transport_plan.items():
        for d in details:
            weekly_loss[k] += d['quantity'] * d['loss_rate']
    
    return {
        'status': LpStatus[prob.status],
        'transport_plan': transport_plan,
        'weekly_transport': weekly_transport,
        'weekly_loss': weekly_loss,
        'total_loss': value(prob.objective)
    }

# ---------------------- 结果可视化 ----------------------
def visualize_results(order_results, transport_results):
    """可视化订购方案和转运方案的结果"""
    print("正在生成可视化结果...")
    
    weeks = 24
    
    # 创建结果保存目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. 每周各类原材料订购量可视化
    plt.figure(figsize=(12, 6))
    week_nums = range(1, weeks+1)
    for t in ['A', 'B', 'C']:
        plt.plot(week_nums, [order_results['weekly_volume'][t][j] for j in week_nums], 
                 marker='o', label=f'{t}类原材料')
    
    plt.title('每周各类原材料订购量')
    plt.xlabel('周数')
    plt.ylabel('订购量 (立方米)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/weekly_volume.png', dpi=300)
    plt.close()
    
    # 2. 库存变化可视化
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, weeks+1), order_results['inventory'], marker='s', color='purple')
    plt.axhline(y=2*28200, color='r', linestyle='--', label='最低库存要求')
    plt.title('每周库存变化')
    plt.xlabel('周数')
    plt.ylabel('库存量 (立方米)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/inventory变化.png', dpi=300)
    plt.close()
    
    # 3. 各转运商每周转运量可视化
    plt.figure(figsize=(14, 8))
    transporters = transport_results['weekly_transport'].keys()
    for j in transporters:
        plt.plot(week_nums, [transport_results['weekly_transport'][j][k] for k in week_nums], 
                 marker='^', label=f'{j}')
    
    plt.axhline(y=6000, color='r', linestyle='--', label='最大转运能力')
    plt.title('各转运商每周转运量')
    plt.xlabel('周数')
    plt.ylabel('转运量 (立方米)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/transporter_weekly.png', dpi=300)
    plt.close()
    
    # 4. 每周损耗量可视化
    plt.figure(figsize=(12, 6))
    plt.bar(week_nums, [transport_results['weekly_loss'][k] for k in week_nums], color='orange')
    plt.title('每周原材料损耗量')
    plt.xlabel('周数')
    plt.ylabel('损耗量 (立方米)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/weekly_loss.png', dpi=300)
    plt.close()
    
    # 5. 原材料类型占比可视化
    total_A = sum(order_results['weekly_volume']['A'][1:weeks+1])
    total_B = sum(order_results['weekly_volume']['B'][1:weeks+1])
    total_C = sum(order_results['weekly_volume']['C'][1:weeks+1])
    
    plt.figure(figsize=(8, 8))
    plt.pie([total_A, total_B, total_C], labels=['A类', 'B类', 'C类'], 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('各类原材料总订购量占比')
    plt.tight_layout()
    plt.savefig('results/materials_pie.png', dpi=300)
    plt.close()
    
    print("可视化结果已保存至results文件夹")

# ---------------------- 结果分析与输出 ----------------------
def analyze_and_export_results(data, order_results, transport_results):
    """分析结果并导出为CSV文件"""
    print("正在分析并导出结果...")
    
    weeks = 24
    suppliers = data['top50_supply']['supplier_ID'].unique()
    transporters = data['transporter_loss']['transporter_ID'].unique()
    
    # 创建结果保存目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. 导出每周订购方案
    order_df = pd.DataFrame(columns=['week'] + list(suppliers))
    order_df['week'] = range(1, weeks+1)
    
    for i in suppliers:
        for j in range(1, weeks+1):
            if (i, j) in order_results['order_plan']:
                order_df.loc[j-1, i] = order_results['order_plan'][(i, j)]['quantity']
            else:
                order_df.loc[j-1, i] = 0
    
    order_df.to_csv('results/订购方案.csv', index=False)
    
    # 2. 导出每周各类材料订购量
    volume_df = pd.DataFrame(columns=['week', 'A类', 'B类', 'C类'])
    volume_df['week'] = range(1, weeks+1)
    
    for j in range(1, weeks+1):
        volume_df.loc[j-1, 'A类'] = order_results['weekly_volume']['A'][j]
        volume_df.loc[j-1, 'B类'] = order_results['weekly_volume']['B'][j]
        volume_df.loc[j-1, 'C类'] = order_results['weekly_volume']['C'][j]
    
    volume_df.to_csv('results/每周材料订购量.csv', index=False)
    
    # 3. 导出库存变化
    inventory_df = pd.DataFrame({
        'week': range(0, weeks+1),
        'inventory': order_results['inventory']
    })
    inventory_df.to_csv('results/库存变化.csv', index=False)
    
    # 4. 导出转运方案
    transport_records = []
    for (i, k), details in transport_results['transport_plan'].items():
        for d in details:
            transport_records.append({
                'supplier': i,
                'week': k,
                'transporter': d['transporter'],
                'quantity': d['quantity'],
                'loss_rate': d['loss_rate'] * 100  # 转换为百分比
            })
    
    transport_df = pd.DataFrame(transport_records)
    transport_df.to_csv('results/转运方案.csv', index=False)
    
    # 5. 导出每周转运量汇总
    transport_summary = pd.DataFrame(columns=['week'] + list(transporters))
    transport_summary['week'] = range(1, weeks+1)
    
    for j in transporters:
        for k in range(1, weeks+1):
            transport_summary.loc[k-1, j] = transport_results['weekly_transport'][j][k]
    
    transport_summary.to_csv('results/每周转运量汇总.csv', index=False)
    
    # 6. 导出每周损耗
    loss_df = pd.DataFrame({
        'week': range(1, weeks+1),
        'loss_quantity': transport_results['weekly_loss'][1:weeks+1]
    })
    loss_df.to_csv('results/每周损耗量.csv', index=False)
    
    # 打印关键结果
    print("\n===== 方案关键指标 =====")
    print(f"总订购量 - A类: {sum(order_results['weekly_volume']['A'][1:weeks+1]):.2f} 立方米")
    print(f"总订购量 - B类: {sum(order_results['weekly_volume']['B'][1:weeks+1]):.2f} 立方米")
    print(f"总订购量 - C类: {sum(order_results['weekly_volume']['C'][1:weeks+1]):.2f} 立方米")
    print(f"A类与C类差值: {order_results['total_A'] - order_results['total_C']:.2f} 立方米")
    print(f"总成本: {order_results['total_cost']:.2f} (相对单位)")
    print(f"总损耗量: {transport_results['total_loss']:.2f} 立方米")
    print(f"平均每周损耗率: {transport_results['total_loss'] / sum(sum(order_results['weekly_volume'][t][1:weeks+1]) for t in ['A','B','C']) * 100:.2f}%")
    
    print("\n结果已导出至results文件夹")

# ---------------------- 主函数 ----------------------
def main():
    """主函数，协调各个步骤"""
    print("===== 原材料订购与转运方案优化 =====")
    
    # 1. 读取数据
    data = load_data()
    
    # 2. 计算供应商供货能力
    supplier_capacity = calculate_supplier_capacity(data)
    
    # 3. 优化订购方案
    order_results = optimize_order_plan(data, supplier_capacity)
    
    # 4. 优化转运方案
    transport_results = optimize_transport_plan(data, order_results['order_plan'])
    
    # 5. 可视化结果
    visualize_results(order_results, transport_results)
    
    # 6. 分析并导出结果
    analyze_and_export_results(data, order_results, transport_results)
    
    print("\n===== 优化完成 =====")

if __name__ == "__main__":
    main()
