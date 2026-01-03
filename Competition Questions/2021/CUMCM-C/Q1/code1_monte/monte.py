import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler
import random

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
def load_and_prepare_data():
    # 读取供货量数据
    supply_data = pd.read_csv('供应商供货量.csv', encoding='utf-8-sig')
    # 读取订货量数据
    order_data = pd.read_csv('企业订货量.csv', encoding='utf-8-sig')
    
    # 清洗列名 - 移除不可见字符
    supply_data.columns = [col.strip() for col in supply_data.columns]
    order_data.columns = [col.strip() for col in order_data.columns]
    
    # 重命名列以保持一致性
    supply_data = supply_data.rename(columns={'供应商ID': '供应商ID', '材料分类': '材料分类'})
    order_data = order_data.rename(columns={'供应商ID': '供应商ID', '材料分类': '材料分类'})
    
    # 清洗供应商ID和材料分类列
    for df in [supply_data, order_data]:
        if '供应商ID' in df.columns:
            df['供应商ID'] = df['供应商ID'].str.strip()
        if '材料分类' in df.columns:
            df['材料分类'] = df['材料分类'].str.strip()
    
    return supply_data, order_data

# 数据预处理
def preprocess_data(supply, order):
    # 合并数据
    merged = pd.merge(supply, order, on=['供应商ID', '材料分类'], suffixes=('_supply', '_order'))
    
    # 获取周次列名
    week_columns = [col for col in merged.columns if re.match(r'^W\d+', col)]
    weeks = [col.replace('_order', '').replace('_supply', '') for col in week_columns]
    weeks = sorted(set(weeks))
    
    # 转换数据格式
    long_format = pd.DataFrame()
    for week in weeks:
        week_supply = week + '_supply'
        week_order = week + '_order'
        
        # 确保列存在
        if week_supply in merged.columns and week_order in merged.columns:
            temp = merged[['供应商ID', '材料分类', week_supply, week_order]].copy()
            temp = temp.rename(columns={week_supply: '供货量', week_order: '订货量'})
            temp['周次'] = week
            long_format = pd.concat([long_format, temp], ignore_index=True)
    
    # 处理异常值：供货量超过订货量3倍时截断
    long_format.loc[long_format['订货量'] > 0, '供货量'] = long_format.apply(
        lambda x: min(x['供货量'], 3 * x['订货量']) if x['订货量'] > 0 else x['供货量'], axis=1)
    
    return long_format

# 计算各项指标
def calculate_metrics(df):
    # 仅考虑订货量>0的记录
    eff_df = df[df['订货量'] > 0].copy()
    
    # 按供应商分组计算
    grouped = eff_df.groupby(['供应商ID', '材料分类'])
    
    # 初始化指标DataFrame
    metrics = grouped.size().reset_index(name='有效周数')
    
    # 计算变异系数 (CV)
    mean_df = grouped['供货量'].mean().reset_index(name='供货量均值')
    std_df = grouped['供货量'].std().reset_index(name='供货量标准差').fillna(0)
    
    metrics = pd.merge(metrics, mean_df, on=['供应商ID', '材料分类'])
    metrics = pd.merge(metrics, std_df, on=['供应商ID', '材料分类'])
    
    metrics['CV'] = (metrics['供货量标准差'] / metrics['供货量均值']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    
    # 计算订单满足波动率 (OFV) - 仅考虑订货量>=10的记录
    ofv_df = eff_df[eff_df['订货量'] >= 10].copy()
    if not ofv_df.empty:
        ofv_df['满足率'] = ofv_df['供货量'] / ofv_df['订货量']
        ofv_grouped = ofv_df.groupby(['供应商ID', '材料分类'])
        
        ofv_mean = ofv_grouped['满足率'].mean().reset_index(name='平均满足率')
        ofv_std = ofv_grouped['满足率'].std().reset_index(name='OFV').fillna(0)
        
        metrics = pd.merge(metrics, ofv_mean, on=['供应商ID', '材料分类'], how='left')
        metrics = pd.merge(metrics, ofv_std, on=['供应商ID', '材料分类'], how='left')
    else:
        metrics['平均满足率'] = 0
        metrics['OFV'] = 0
    
    # 计算订单满足率 (OFR)
    ofr_df = grouped.apply(
        lambda x: pd.Series({
            'OFR': np.minimum(x['供货量'], x['订货量']).sum() / x['订货量'].sum() * 100
        })
    ).reset_index()
    metrics = pd.merge(metrics, ofr_df, on=['供应商ID', '材料分类'])
    
    # 计算缺货风险指数 (SRI)
    eff_df['缺货'] = (eff_df['供货量'] < 0.9 * eff_df['订货量']).astype(int)
    
    shortage_count = eff_df.groupby(['供应商ID', '材料分类'])['缺货'].sum().reset_index(name='缺货周数')
    shortage_volume = eff_df.groupby(['供应商ID', '材料分类']).apply(
        lambda x: (x['订货量'] - x['供货量']).clip(lower=0).sum()
    ).reset_index(name='总缺货量')
    order_volume = eff_df.groupby(['供应商ID', '材料分类'])['订货量'].sum().reset_index(name='总订货量')
    
    metrics = pd.merge(metrics, shortage_count, on=['供应商ID', '材料分类'])
    metrics = pd.merge(metrics, shortage_volume, on=['供应商ID', '材料分类'])
    metrics = pd.merge(metrics, order_volume, on=['供应商ID', '材料分类'])
    
    metrics['SRI'] = (metrics['缺货周数'] / metrics['有效周数']) * (metrics['总缺货量'] / metrics['总订货量']) * 100
    
    # 计算持续供货能力 (CA)
    all_df = df.copy()
    all_df['有效供货'] = ((all_df['订货量'] > 0) & (all_df['供货量'] > 0)) | ((all_df['订货量'] == 0) & (all_df['供货量'] > 0))
    
    effective_weeks = all_df.groupby(['供应商ID', '材料分类'])['有效供货'].sum().reset_index(name='有效供货周数')
    total_weeks = all_df.groupby(['供应商ID', '材料分类']).size().reset_index(name='总周数')
    
    metrics = pd.merge(metrics, effective_weeks, on=['供应商ID', '材料分类'])
    metrics = pd.merge(metrics, total_weeks, on=['供应商ID', '材料分类'])
    
    metrics['CA'] = (metrics['有效供货周数'] / metrics['总周数']) * 100
    
    # 计算产能贡献指数 (PCI)
    metrics['PCI'] = metrics['供货量均值']
    metrics.loc[metrics['材料分类'] == 'A', 'PCI'] *= 1.67
    metrics.loc[metrics['材料分类'] == 'B', 'PCI'] *= 1.52
    metrics.loc[metrics['材料分类'] == 'C', 'PCI'] *= 1.39
    
    # 筛选有效周数>=3的供应商
    metrics = metrics[metrics['有效周数'] >= 3]
    
    # 填充可能的NaN值
    metrics.fillna(0, inplace=True)
    
    # 删除中间列
    metrics = metrics.drop(['供货量标准差', '总缺货量', '总订货量', '有效供货周数', '总周数'], axis=1, errors='ignore')
    
    return metrics.reset_index(drop=True)

# TOPSIS评价模型
def topsis_evaluation(data, weights, indicators):
    # 提取指标数据
    indicator_data = data[indicators].copy()
    
    # 标准化处理 - 分正向和负向指标
    normalized = pd.DataFrame()
    for col in indicators:
        col_min = indicator_data[col].min()
        col_max = indicator_data[col].max()
        range_val = col_max - col_min
        
        # 避免除零错误
        if range_val == 0:
            normalized[col] = 0.5  # 所有值相等时取中值
        else:
            if col in ['CV', 'OFV', 'SRI']:  # 负向指标
                normalized[col] = (col_max - indicator_data[col]) / range_val
            else:  # 正向指标
                normalized[col] = (indicator_data[col] - col_min) / range_val
    
    # 应用权重
    weighted = normalized * weights
    
    # 确定正负理想解
    positive_ideal = weighted.max()
    negative_ideal = weighted.min()
    
    # 计算距离
    pos_distance = np.sqrt(((weighted - positive_ideal) ** 2).sum(axis=1))
    neg_distance = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    
    # 计算综合得分
    scores = neg_distance / (pos_distance + neg_distance)
    
    return scores

# 蒙特卡洛敏感性分析
def monte_carlo_sensitivity(metrics_df, weights, indicators, initial_top50, 
                           n_simulations=200, perturbation=0.05):
    """
    执行蒙特卡洛敏感性分析
    
    参数:
    metrics_df -- 包含所有供应商指标的DataFrame
    weights -- 初始权重数组
    indicators -- 指标名称列表
    initial_top50 -- 初始Top50供应商
    n_simulations -- 模拟次数，默认为200
    perturbation -- 权重扰动范围，默认为±5%
    
    返回:
    frequency_df -- 每个供应商出现在Top50的频率
    stability_df -- Top50供应商的稳定性数据
    avg_overlap_rate -- 平均重叠率
    """
    # 初始化频率计数器
    supplier_ids = metrics_df['供应商ID'].unique()
    frequency = {id: 0 for id in supplier_ids}
    
    # 存储每次模拟的Top50
    all_top50 = []
    
    # 初始Top50的供应商ID列表
    initial_top50_ids = initial_top50['供应商ID'].tolist()
    
    # 运行蒙特卡洛模拟
    for i in range(n_simulations):
        # 复制原始指标数据，避免修改原始数据
        current_metrics = metrics_df.copy()
        
        # 生成扰动权重
        perturbed_weights = weights * (1 + perturbation * (2 * np.random.random(len(weights)) - 1))
        
        # 归一化权重（总和为1）
        perturbed_weights /= perturbed_weights.sum()
        
        # 使用扰动权重重新计算综合得分
        current_metrics['综合得分'] = topsis_evaluation(current_metrics, perturbed_weights, indicators)
        
        # 按综合得分排序并选择Top50
        top_50 = current_metrics.sort_values('综合得分', ascending=False).head(50)
        top_50_ids = top_50['供应商ID'].tolist()
        all_top50.append(top_50_ids)
        
        # 更新频率计数器
        for id in top_50_ids:
            frequency[id] += 1
    
    # 计算频率
    frequency_df = pd.DataFrame({
        '供应商ID': list(frequency.keys()),
        '出现频率': [freq / n_simulations for freq in frequency.values()]
    })
    
    # 计算稳定性指标
    stability_df = pd.DataFrame({
        '供应商ID': metrics_df['供应商ID'],
        '平均排名': np.zeros(len(metrics_df)),
        '排名标准差': np.zeros(len(metrics_df)),
        '排名变异系数': np.zeros(len(metrics_df))
    }).set_index('供应商ID')
    
    # 计算每个供应商的平均排名和排名波动性
    total_suppliers = len(metrics_df)
    for supplier_id in supplier_ids:
        ranks = []
        for top50 in all_top50:
            if supplier_id in top50:
                # 获取排名（索引从0开始，排名从1开始）
                rank = top50.index(supplier_id) + 1
                ranks.append(rank)
            else:
                # 不在Top50中则赋予一个较大排名（总供应商数+1）
                ranks.append(total_suppliers + 1)
        
        avg_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        cv_rank = std_rank / avg_rank if avg_rank > 0 else 0
        
        stability_df.loc[supplier_id, '平均排名'] = avg_rank
        stability_df.loc[supplier_id, '排名标准差'] = std_rank
        stability_df.loc[supplier_id, '排名变异系数'] = cv_rank
    
    # 按平均排名排序
    stability_df = stability_df.sort_values('平均排名').reset_index()
    
    # 计算平均重叠率
    overlap_rates = []
    for top50 in all_top50:
        overlap_count = len(set(top50) & set(initial_top50_ids))
        overlap_rates.append(overlap_count / 50)
    
    avg_overlap_rate = np.mean(overlap_rates)
    
    return frequency_df, stability_df, avg_overlap_rate

# 可视化敏感性分析结果
def plot_sensitivity_results(metrics_df, frequency_df, stability_df, initial_top50, indicators, initial_weights):
    
    # 1. Top50供应商出现频率分布 
    plt.figure(figsize=(10, 12))
    # 只关注原始Top50供应商的频率
    top50_freq = frequency_df[frequency_df['供应商ID'].isin(initial_top50['供应商ID'])]
    top50_freq = top50_freq.sort_values('出现频率', ascending=False)
    
    plt.barh(top50_freq['供应商ID'], top50_freq['出现频率'], color='dodgerblue')
    plt.axvline(x=0.9, color='red', linestyle='--', alpha=0.7)
    plt.title('Top50供应商出现频率')
    plt.xlabel('出现频率')
    plt.ylabel('供应商ID')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.savefig('Top50供应商出现频率.png',dpi=300)
    plt.show()
    
    # 2. 频率分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(frequency_df['出现频率'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    plt.title('供应商出现频率分布')
    plt.xlabel('出现频率')
    plt.ylabel('供应商数量')
    plt.grid(alpha=0.3)
    plt.savefig('出现频率分布.png',dpi=300)
    plt.show()
    
    # 3. 稳定性分析 - 平均排名 vs 排名变异系数
    plt.figure(figsize=(10, 6))
    stability_top50 = stability_df[stability_df['供应商ID'].isin(initial_top50['供应商ID'])]
    
    plt.scatter(stability_top50['平均排名'], stability_top50['排名变异系数'], 
                s=100, c='coral', alpha=0.7, edgecolor='black')
    
    # 添加标签
    for i, row in stability_top50.iterrows():
        if row['平均排名'] < 20 or row['排名变异系数'] > 0.5:  # 只标记重要点
            plt.annotate(row['供应商ID'], (row['平均排名'], row['排名变异系数']), 
                         xytext=(5, 5), textcoords='offset points')
    
    plt.title('Top50供应商排名稳定性')
    plt.xlabel('平均排名')
    plt.ylabel('排名变异系数')
    plt.grid(alpha=0.3)
    plt.savefig('Top50供应商排名稳定性.png',dpi=300)
    plt.show()
    
    # 4. 权重敏感性分析
    # 模拟权重变化的影响
    weight_impact = []
    
    for i in range(len(initial_weights)):
        weight_changes = np.linspace(0.5 * initial_weights[i], 1.5 * initial_weights[i], 10)
        
        for w in weight_changes:
            current_metrics = metrics_df.copy()
            new_weights = initial_weights.copy()
            new_weights[i] = w
            new_weights /= new_weights.sum()  # 归一化
            
            # 计算综合得分
            current_metrics['综合得分'] = topsis_evaluation(current_metrics, new_weights, indicators)
            top_50 = current_metrics.sort_values('综合得分', ascending=False).head(50)
            
            # 计算与原始Top50的重叠率
            overlap = len(set(initial_top50['供应商ID']) & set(top_50['供应商ID'])) / 50
            weight_impact.append({
                '指标': indicators[i],
                '权重变化': w / initial_weights[i],
                'Top50重叠率': overlap
            })
    
    weight_impact_df = pd.DataFrame(weight_impact)
    
    plt.figure(figsize=(10, 6))
    for indicator in indicators:
        subset = weight_impact_df[weight_impact_df['指标'] == indicator]
        plt.plot(subset['权重变化'], subset['Top50重叠率'], 
                 marker='o', label=indicator, linewidth=2)
    
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    plt.title('权重变化对Top50重叠率的影响')
    plt.xlabel('权重变化倍数')
    plt.ylabel('Top50重叠率')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('权重变化对Top50重叠率的影响.png',dpi=300)
    plt.show()
    
    # 5. 指标权重重要性
    plt.figure(figsize=(10, 6))
    weight_importance = []
    for indicator in indicators:
        # 固定其他权重，变化当前指标权重
        base_weights = initial_weights.copy()
        idx = indicators.index(indicator)
        
        # 计算权重变化时的平均重叠率
        changes = weight_impact_df[weight_impact_df['指标'] == indicator]
        sensitivity = np.mean(np.abs(np.diff(changes['Top50重叠率'])))
        weight_importance.append(sensitivity)
    
    plt.barh(indicators, weight_importance, color='mediumseagreen')
    plt.title('指标权重敏感性')
    plt.xlabel('敏感性指数')
    plt.grid(axis='x', alpha=0.3)
    plt.savefig('指标权重敏感性.png',dpi=300)
    plt.show()
    
    # 6. Top50供应商稳定性热图
    plt.figure(figsize=(10, 6))
    # 选择最稳定的10家供应商
    stable_suppliers = stability_df.sort_values('平均排名').head(10)['供应商ID']
    stable_data = []
    
    # 模拟10次不同权重下的排名
    for i in range(10):
        current_metrics = metrics_df.copy()
        perturbed_weights = initial_weights * (1 + 0.1 * (2 * np.random.random(len(initial_weights)) - 1))
        perturbed_weights /= perturbed_weights.sum()
        
        current_metrics['综合得分'] = topsis_evaluation(current_metrics, perturbed_weights, indicators)
        metrics_sorted = current_metrics.sort_values('综合得分', ascending=False)
        
        # 获取供应商排名
        ranks = []
        for supplier in stable_suppliers:
            rank = metrics_sorted.index[metrics_sorted['供应商ID'] == supplier][0] + 1
            ranks.append(rank)
        
        stable_data.append(ranks)
    
    stable_df = pd.DataFrame(stable_data, columns=stable_suppliers).T
    
    sns.heatmap(stable_df, cmap='YlGnBu', annot=True, fmt="d", 
                cbar_kws={'label': '排名'})
    plt.title('稳定供应商在不同权重下的排名')
    plt.xlabel('模拟次数')
    plt.ylabel('供应商ID')
    
    plt.tight_layout()
    plt.savefig('稳定供应商在不同权重下的排名.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 额外图表：初始Top50与高频率供应商的关系
    plt.figure(figsize=(12, 8))
    
    # 合并初始排名和频率
    merged = initial_top50.merge(frequency_df, on='供应商ID')
    merged = merged.merge(stability_df, on='供应商ID')
    
    plt.scatter(merged['综合得分'], merged['出现频率'], 
                c=merged['平均排名'], cmap='viridis_r', s=100)
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('平均排名')
    
    # 添加标签
    for i, row in merged.iterrows():
        if row['出现频率'] < 0.7 or row['综合得分'] > 0.9:  # 只标记重要点
            plt.annotate(row['供应商ID'], (row['综合得分'], row['出现频率']), 
                         xytext=(5, 5), textcoords='offset points')
    
    plt.title('初始得分 vs 出现频率 (颜色表示平均排名)')
    plt.xlabel('初始综合得分')
    plt.ylabel('蒙特卡洛模拟出现频率')
    plt.grid(alpha=0.3)
    plt.savefig('初始得分VS出现频率.png', dpi=300)
    plt.show()

# 主函数
def main():
    # 加载数据
    supply_data, order_data = load_and_prepare_data()
    
    # 预处理数据
    long_data = preprocess_data(supply_data, order_data)
    
    if long_data.empty:
        print("数据预处理后为空，请检查数据格式和列名")
        return
    
    # 计算指标
    metrics_df = calculate_metrics(long_data)
    
    if metrics_df.empty:
        print("指标计算后为空，请检查数据")
        return
    
    # 设置指标和权重
    indicators = ['CV', 'OFV', 'OFR', 'SRI', 'CA', 'PCI']
    initial_weights = np.array([0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
    
    # 使用初始权重计算综合得分
    metrics_df['综合得分'] = topsis_evaluation(metrics_df, initial_weights, indicators)
    
    # 保存所有供应商指标数据
    metrics_df.to_csv('supplier_metrics.csv', index=False)
    print("供应商指标数据已保存到 supplier_metrics.csv")
    
   # 获取初始Top50供应商
    initial_top50 = metrics_df.sort_values('综合得分', ascending=False).head(50)
    
    # 执行蒙特卡洛敏感性分析
    print("开始进行敏感性分析...")
    frequency_df, stability_df, avg_overlap_rate = monte_carlo_sensitivity(
        metrics_df, initial_weights, indicators, initial_top50,
        n_simulations=500, perturbation=0.05
    )
    
    # 保存结果
    frequency_df.to_csv('supplier_frequency.csv', index=False)
    stability_df.to_csv('supplier_stability.csv', index=False)
    print("敏感性分析结果已保存")
    
    # 可视化结果
    plot_sensitivity_results(metrics_df, frequency_df, stability_df, initial_top50, indicators, initial_weights)
    
    # 分析结果
    print("\n敏感性分析完成！")
    print(f"平均Top50重叠率: {avg_overlap_rate:.2%}")  # 使用计算出的重叠率
    
    # 识别最稳定的供应商
    stable_suppliers = stability_df.sort_values('平均排名').head(10)
    print("\n最稳定的10家供应商:")
    print(stable_suppliers[['供应商ID', '平均排名', '排名变异系数']])
    
    # 识别权重最敏感的指标
    weight_importance = []
    for indicator in indicators:
        changes = np.linspace(0.5, 1.5, 10)
        overlaps = []
        
        for c in changes:
            current_metrics = metrics_df.copy()
            new_weights = initial_weights.copy()
            idx = indicators.index(indicator)
            new_weights[idx] = initial_weights[idx] * c
            new_weights /= new_weights.sum()
            
            current_metrics['综合得分'] = topsis_evaluation(current_metrics, new_weights, indicators)
            top_50 = current_metrics.sort_values('综合得分', ascending=False).head(50)
            overlap = len(set(initial_top50['供应商ID']) & set(top_50['供应商ID'])) / 50
            overlaps.append(overlap)
        
        sensitivity = np.mean(np.abs(np.diff(overlaps)))
        weight_importance.append(sensitivity)
    
    sensitivity_df = pd.DataFrame({
        '指标': indicators,
        '敏感性': weight_importance
    }).sort_values('敏感性', ascending=False)
    
    print("\n指标权重敏感性排序:")
    print(sensitivity_df)
    
    # 识别核心供应商
    core_suppliers = frequency_df[frequency_df['出现频率'] > 0.9]
    print(f"\n核心供应商（出现频率>90%）: {len(core_suppliers)}家")
    print(core_suppliers.sort_values('出现频率', ascending=False))

# 执行主函数
if __name__ == "__main__":
    main()