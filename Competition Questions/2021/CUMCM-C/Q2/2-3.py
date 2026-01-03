import numpy as np
import pandas as pd
import pulp as pl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class MaterialOptimizer:
    def __init__(self, weeks=24):
        """初始化优化器参数，使用实际数据"""
        self.weeks = weeks  # 计划周期
        
        # 材料类型参数 (A, B, C)
        self.material_types = ['A', 'B', 'C']
        self.G = {'A': 1.2, 'B': 1.1, 'C': 1.0}  # 采购单价相对值
        self.f = {'A': 1/0.6, 'B': 1/0.66, 'C': 1/0.72}  # 单位材料产出系数
        self.P = 28200  # 每周产能需求（立方米）
        self.H = 0.1  # 单位存储成本（相对于C材料采购价的比例）
        
        # 读取实际数据
        self.supplier_data = None           # 供应商数据
        self.transporter_loss_data = None   # 转运商损耗率数据
        self.supplier_materials = {}        # 供应商-材料类型映射
        self.max_supply = None              # 供应商最大供货量
        self.loss_rates = None              # 转运商损耗率
        
        # 存储结果
        self.order_plan = None
        self.transport_plan = None
        self.total_cost = 0
        self.total_loss = 0
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载实际数据"""
        print("正在加载实际数据...")
        
        # 读取Top17供应商数据
        self.supplier_data = pd.read_csv("data/Top17供应商供货量.csv")
        self.top_suppliers = self.supplier_data['supplier_ID'].unique()
        self.top_n_suppliers = len(self.top_suppliers)  # 获取实际供应商数量
        
        # 建立供应商-材料类型映射
        for idx, row in self.supplier_data.iterrows():
            self.supplier_materials[row['supplier_ID']] = row['material']
        
        # 计算前10个周期中每周的最大供货量作为约束
        # 假设W001-W010是前10个周期，用于计算最大供货量
        week_columns = [f'W{i:03d}' for i in range(1, 11)]  # W001到W010
        
        # 初始化最大供货量矩阵 (供应商数 x 周数)
        self.max_supply = pd.DataFrame(
            index=self.top_suppliers,
            columns=[f'W{i:03d}' for i in range(1, self.weeks+1)]  # 未来24周 W001-W024
        )
        
        # 对未来24周的每一周，使用历史同期的最大供货量作为约束
        # 假设历史数据足够多，可以按周循环使用
        for week in range(1, self.weeks+1):
            week_col = f'W{week:03d}'
            # 找到历史数据中对应周数的列（循环使用前10周数据）
            hist_week = ((week - 1) % 10) + 1
            hist_week_col = f'W{hist_week:03d}'
            
            for supplier in self.top_suppliers:
                # 获取该供应商在历史同期的最大供货量
                supplier_data = self.supplier_data[self.supplier_data['supplier_ID'] == supplier]
                self.max_supply.loc[supplier, week_col] = supplier_data[hist_week_col].max()
        
        # 读取转运商损耗率数据
        self.transporter_loss_data = pd.read_csv("data/转运商运输损耗率.csv")
        self.transporters = self.transporter_loss_data['transporter_ID'].unique()
        self.num_transporters = len(self.transporters)
        
        # 转换损耗率数据为矩阵 (转运商数 x 周数)
        self.loss_rates = pd.DataFrame(
            index=self.transporters,
            columns=[f'W{i:03d}' for i in range(1, self.weeks+1)]
        )
        
        for week in range(1, self.weeks+1):
            week_col = f'W{week:03d}'
            # 从历史数据中获取损耗率，假设直接使用对应周的数据
            for transporter in self.transporters:
                trans_data = self.transporter_loss_data[
                    self.transporter_loss_data['transporter_ID'] == transporter
                ]
                # 转换为小数形式的损耗率
                self.loss_rates.loc[transporter, week_col] = float(trans_data[week_col].values[0]) / 100
        
        # 将数据转换为numpy数组以便处理，同时建立索引映射
        self.supplier_index = {supplier: i for i, supplier in enumerate(self.top_suppliers)}
        self.transporter_index = {transporter: i for i, transporter in enumerate(self.transporters)}
        
        # 转换最大供货量为numpy数组
        self.max_supply_np = np.zeros((self.top_n_suppliers, self.weeks))
        for i, supplier in enumerate(self.top_suppliers):
            for j in range(self.weeks):
                week_col = f'W{j+1:03d}'
                self.max_supply_np[i, j] = self.max_supply.loc[supplier, week_col]
        
        # 转换损耗率为numpy数组
        self.loss_rates_np = np.zeros((self.num_transporters, self.weeks))
        for i, transporter in enumerate(self.transporters):
            for j in range(self.weeks):
                week_col = f'W{j+1:03d}'
                self.loss_rates_np[i, j] = self.loss_rates.loc[transporter, week_col]
        
        print(f"数据加载完成。使用{self.top_n_suppliers}家供应商和{self.num_transporters}家转运商的数据")
    
    def optimize_order_plan(self):
        """优化订购方案"""
        print("开始优化订购方案...")
        
        # 创建问题实例，最小化成本
        prob = pl.LpProblem("Order_Optimization", pl.LpMinimize)
        
        # 定义决策变量：x[i,j] 第i个供应商第j周的订货量
        x = pl.LpVariable.dicts("x", 
                               (range(self.top_n_suppliers), range(self.weeks)),
                               lowBound=0, 
                               cat='Continuous')
        
        # 目标函数：最小化总采购和存储成本
        prob += pl.lpSum([
            (self.G[self.supplier_materials[self.top_suppliers[i]]] + self.H) * x[i][j] 
            for i in range(self.top_n_suppliers) 
            for j in range(self.weeks)
        ]), "Total_Cost"
        
        # 约束条件1：每周总产能满足需求
        for j in range(self.weeks):
            prob += pl.lpSum([
                x[i][j] * self.f[self.supplier_materials[self.top_suppliers[i]]] 
                for i in range(self.top_n_suppliers)
            ]) >= self.P, f"Capacity_Constraint_Week_{j}"
        
        # 约束条件2：不超过最大供货量
        for i in range(self.top_n_suppliers):
            for j in range(self.weeks):
                prob += x[i][j] <= self.max_supply_np[i, j], \
                        f"Max_Supply_Constraint_Supplier_{i}_Week_{j}"
        
        # 求解问题
        prob.solve(pl.PULP_CBC_CMD(msg=0))  # 静默模式求解
        
        # 检查求解状态
        if pl.LpStatus[prob.status] != "Optimal":
            print(f"警告：订购方案求解未找到最优解，状态为：{pl.LpStatus[prob.status]}")
        
        # 保存结果
        self.order_plan = np.zeros((self.top_n_suppliers, self.weeks))
        for i in range(self.top_n_suppliers):
            for j in range(self.weeks):
                self.order_plan[i, j] = x[i][j].varValue
        
        self.total_cost = prob.objective.value()
        print(f"订购方案优化完成，总成本: {self.total_cost:.2f}")
        
        return self.order_plan
    
    def optimize_transport_plan(self):
        """优化转运方案"""
        if self.order_plan is None:
            print("请先优化订购方案")
            return None
        
        print("开始优化转运方案...")
        
        # 创建问题实例，最小化损耗
        prob = pl.LpProblem("Transport_Optimization", pl.LpMinimize)
        
        # 定义决策变量：x[i,j,k] 第k周，第i个供应商托运给第j个转运商的货量
        x = pl.LpVariable.dicts("x", 
                               (range(self.top_n_suppliers), 
                                range(self.num_transporters), 
                                range(self.weeks)),
                               lowBound=0, 
                               cat='Continuous')
        
        # 目标函数：最小化总损耗
        prob += pl.lpSum([
            x[i][j][k] * self.loss_rates_np[j, k] 
            for i in range(self.top_n_suppliers)
            for j in range(self.num_transporters)
            for k in range(self.weeks)
        ]), "Total_Loss"
        
        # 约束条件1：转运商每周转运量不超过6000
        for j in range(self.num_transporters):
            for k in range(self.weeks):
                prob += pl.lpSum([x[i][j][k] for i in range(self.top_n_suppliers)]) <= 6000, \
                        f"Transporter_Capacity_Constraint_{j}_Week_{k}"
        
        # 约束条件2：转运量等于订货量
        for i in range(self.top_n_suppliers):
            for k in range(self.weeks):
                prob += pl.lpSum([x[i][j][k] for j in range(self.num_transporters)]) == self.order_plan[i, k], \
                        f"Supply_Demand_Constraint_{i}_Week_{k}"
        
        # 求解问题
        prob.solve(pl.PULP_CBC_CMD(msg=0))  # 静默模式求解
        
        # 检查求解状态
        if pl.LpStatus[prob.status] != "Optimal":
            print(f"警告：转运方案求解未找到最优解，状态为：{pl.LpStatus[prob.status]}")
        
        # 保存结果
        self.transport_plan = np.zeros((self.top_n_suppliers, self.num_transporters, self.weeks))
        for i in range(self.top_n_suppliers):
            for j in range(self.num_transporters):
                for k in range(self.weeks):
                    self.transport_plan[i, j, k] = x[i][j][k].varValue
        
        self.total_loss = prob.objective.value()
        print(f"转运方案优化完成，总损耗量: {self.total_loss:.2f}")
        
        return self.transport_plan
    
    def visualize_order_plan(self):
        """可视化订购方案"""
        if self.order_plan is None:
            print("请先优化订购方案")
            return
        
        # 1. 每周总订货量
        weekly_total = np.sum(self.order_plan, axis=0)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=range(1, self.weeks+1), y=weekly_total)
        plt.title('每周总订货量')
        plt.xlabel('周')
        plt.ylabel('订货量 (立方米)')
        plt.axhline(y=self.P / max(self.f.values()), color='r', linestyle='--', 
                   label=f'最小需求参考线')
        plt.legend()
        plt.tight_layout()
        plt.savefig("每周总订货量.png",dpi=300)
        plt.show()
        
        
        # 2. 各供应商总订货量
        supplier_total = np.sum(self.order_plan, axis=1)
        supplier_materials = [self.supplier_materials[supplier] for supplier in self.top_suppliers]
        
        plt.figure(figsize=(12, 6))
        colors = {'A': 'red', 'B': 'green', 'C': 'blue'}
        bars = plt.bar(range(1, self.top_n_suppliers+1), supplier_total)
        for i, bar in enumerate(bars):
            bar.set_color(colors[supplier_materials[i]])
        
        # 添加图例
        for mat in self.material_types:
            plt.bar(0, 0, color=colors[mat], label=mat)
        
        plt.title('各供应商总订货量')
        plt.xlabel('供应商编号')
        plt.ylabel('总订货量 (立方米)')
        plt.legend(title='材料类型')
        plt.tight_layout()
        plt.savefig("各供应商总订货量.png",dpi=300)
        plt.show()
        
        
        # 3. 各材料类型每周订货量
        mat_weekly = {mat: np.zeros(self.weeks) for mat in self.material_types}
        for i in range(self.top_n_suppliers):
            mat = self.supplier_materials[self.top_suppliers[i]]
            mat_weekly[mat] += self.order_plan[i, :]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(1, self.weeks+1)
        width = 0.25
        for i, mat in enumerate(self.material_types):
            plt.bar(x + i*width, mat_weekly[mat], width=width, label=mat)
        
        plt.title('各材料类型每周订货量')
        plt.xlabel('周')
        plt.ylabel('订货量 (立方米)')
        plt.xticks(x + width, x)
        plt.legend(title='材料类型')
        plt.tight_layout()
        plt.savefig("各材料类型每周订货量.png",dpi=300)
        plt.show()
    
    def visualize_transport_plan(self):
        """可视化转运方案"""
        if self.transport_plan is None:
            print("请先优化转运方案")
            return
        
        # 1. 各转运商每周运输量
        transporter_weekly = np.sum(self.transport_plan, axis=0)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(1, self.weeks+1)
        width = 0.1
        for j in range(self.num_transporters):
            plt.bar(x + j*width, transporter_weekly[j, :], width=width, label=f'{self.transporters[j]}')
        
        plt.axhline(y=6000, color='r', linestyle='--', label='最大运输能力')
        plt.title('各转运商每周运输量')
        plt.xlabel('周')
        plt.ylabel('运输量 (立方米)')
        plt.xticks(x + width*(self.num_transporters/2 - 0.5), x)
        plt.legend(title='转运商', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("各转运商每周运输量.png",dpi=300)
        plt.show()
        
        # 2. 各转运商总运输量和平均损耗率
        transporter_total = np.sum(transporter_weekly, axis=1)
        transporter_avg_loss = np.mean(self.loss_rates_np, axis=1) * 100  # 转换为百分比
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('转运商')
        ax1.set_ylabel('总运输量 (立方米)', color=color)
        ax1.bar(self.transporters, transporter_total, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  # 创建第二个y轴
        color = 'tab:red'
        ax2.set_ylabel('平均损耗率 (%)', color=color)
        ax2.plot(self.transporters, transporter_avg_loss, 
                 color=color, marker='o', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('各转运商总运输量和平均损耗率')
        fig.tight_layout()
        plt.savefig("各转运商总运输量和平均损耗率.png",dpi=300)
        plt.show()
        
        
        # 3. 每周总损耗量
        weekly_loss = np.sum(np.sum(self.transport_plan * self.loss_rates_np[np.newaxis, :, :], 
                                    axis=0), axis=0)
        weekly_total = np.sum(self.order_plan, axis=0)
        weekly_loss_rate = (weekly_loss / weekly_total) * 100  # 转换为百分比
        
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        color = 'tab:blue'
        ax1.set_xlabel('周')
        ax1.set_ylabel('总损耗量 (立方米)', color=color)
        ax1.bar(range(1, self.weeks+1), weekly_loss, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('损耗率 (%)', color=color)
        ax2.plot(range(1, self.weeks+1), weekly_loss_rate, color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('每周总损耗量和损耗率')
        plt.tight_layout()
        plt.savefig("每周总损耗量和损耗率.png",dpi=300)
        plt.show()
        
    
    def analyze_results(self):
        """分析优化结果"""
        if self.order_plan is None or self.transport_plan is None:
            print("请先完成订购方案和转运方案的优化")
            return
        
        # 计算各材料类型的总采购量和占比
        mat_total = {mat: 0 for mat in self.material_types}
        for i in range(self.top_n_suppliers):
            mat = self.supplier_materials[self.top_suppliers[i]]
            mat_total[mat] += np.sum(self.order_plan[i, :])
        
        total_order = np.sum(self.order_plan)
        mat_percent = {mat: (mat_total[mat]/total_order)*100 for mat in self.material_types}
        
        print("\n=== 订购方案分析 ===")
        print(f"总订货量: {total_order:.2f} 立方米")
        print("各材料类型占比:")
        for mat, percent in mat_percent.items():
            print(f"  {mat}类材料: {percent:.2f}% ({mat_total[mat]:.2f} 立方米)")
        print(f"总成本: {self.total_cost:.2f} (相对单位)")
        
        # 计算总损耗率
        total_loss_rate = (self.total_loss / total_order) * 100
        
        print("\n=== 转运方案分析 ===")
        print(f"总运输量: {total_order:.2f} 立方米")
        print(f"总损耗量: {self.total_loss:.2f} 立方米")
        print(f"总损耗率: {total_loss_rate:.2f}%")
        
        # 计算各转运商的运输量占比
        transporter_total = np.sum(np.sum(self.transport_plan, axis=0), axis=1)
        print("\n各转运商运输量占比:")
        for j in range(self.num_transporters):
            percent = (transporter_total[j]/total_order)*100
            print(f"  转运商{self.transporters[j]}: {percent:.2f}% ({transporter_total[j]:.2f} 立方米)")
    
    def export_results(self):
        """导出优化结果为CSV文件"""
        if self.order_plan is None or self.transport_plan is None:
            print("请先完成订购方案和转运方案的优化")
            return
        
        # 导出订购方案
        order_df = pd.DataFrame(columns=['supplier_ID', 'material'] + [f'W{i:03d}' for i in range(1, self.weeks+1)])
        for i, supplier in enumerate(self.top_suppliers):
            row = {
                'supplier_ID': supplier,
                'material': self.supplier_materials[supplier]
            }
            for j in range(self.weeks):
                row[f'W{j+1:03d}'] = self.order_plan[i, j]
            order_df = pd.concat([order_df, pd.DataFrame([row])], ignore_index=True)
        
        order_df.to_csv('优化后的订购方案.csv', index=False)
        print("订购方案已导出至 '优化后的订购方案.csv'")
        
        # 导出转运方案
        transport_df = pd.DataFrame(columns=['supplier_ID', 'transporter_ID'] + [f'W{i:03d}' for i in range(1, self.weeks+1)])
        for i, supplier in enumerate(self.top_suppliers):
            for j, transporter in enumerate(self.transporters):
                row = {
                    'supplier_ID': supplier,
                    'transporter_ID': transporter
                }
                for k in range(self.weeks):
                    row[f'W{k+1:03d}'] = self.transport_plan[i, j, k]
                transport_df = pd.concat([transport_df, pd.DataFrame([row])], ignore_index=True)
        
        transport_df.to_csv('优化后的转运方案.csv', index=False)
        print("转运方案已导出至 '优化后的转运方案.csv'")

# 主函数
def main():
    # 创建优化器实例，计划周期24周
    optimizer = MaterialOptimizer(weeks=24)
    
    # 优化订购方案
    order_plan = optimizer.optimize_order_plan()
    
    # 优化转运方案
    transport_plan = optimizer.optimize_transport_plan()
    
    # 分析结果
    optimizer.analyze_results()
    
    # 可视化结果
    optimizer.visualize_order_plan()
    optimizer.visualize_transport_plan()
    
    # 导出结果
    optimizer.export_results()

if __name__ == "__main__":
    main()
    