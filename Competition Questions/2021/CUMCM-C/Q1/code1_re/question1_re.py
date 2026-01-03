import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler

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

# 单独图表函数
def plot_separate_charts(metrics, top_50):
    """多张图表输出"""
    # 1. CV分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics['CV'], bins=20, kde=True, color='skyblue')
    plt.title('供应商变异系数(CV)分布')
    plt.xlabel('变异系数(CV)值(%)')
    plt.ylabel('供应商数量')
    plt.tight_layout()
    plt.savefig('CV_供应商变异系数.png', dpi=300)
    plt.show()
    
    # 2. OFR分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics['OFR'], bins=20, kde=True, color='salmon')
    plt.title('供应商订单满足率(OFR)分布')
    plt.xlabel('订单满足率(OFR)值(%)')
    plt.ylabel('供应商数量')
    plt.tight_layout()
    plt.savefig('OFR_供应商订单满足率分布.png', dpi=300)
    plt.show()
    
    # 3. 材料分类占比饼图
    if not top_50.empty:
        plt.figure(figsize=(10, 6))
        material_top = top_50['材料分类'].value_counts()
        plt.pie(material_top, labels=material_top.index, autopct='%1.1f%%', 
                colors=['#66c2a5', '#fc8d62', '#8da0cb'], shadow=True,
                startangle=90, explode=(0.05, 0.05, 0.05))
        plt.title('Top50供应商材料分类占比')
        plt.tight_layout()
        plt.savefig('Top50供应商材料分类占比.png', dpi=300)
        plt.show()
    
    # 4. 指标相关性热力图
    if not metrics.empty:
        plt.figure(figsize=(10, 8))
        corr_matrix = metrics[['CV', 'OFV', 'OFR', 'SRI', 'CA', 'PCI']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0,
                   annot_kws={'size': 12}, cbar_kws={'shrink': 0.8})
        plt.title('供应商评价指标间相关系数热力图', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('供应商评价指标间相关系数热力图.png', dpi=300)
        plt.show()
    
    # 5. Top10供应商指标雷达图
    if len(top_50) >= 10:
        top10 = top_50.head(10)
        indicators = ['CV', 'OFV', 'OFR', 'SRI', 'CA', 'PCI']
        n_indicators = len(indicators)
        
        angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # 标准化指标数据
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(top10[indicators])
        
        for i in range(len(top10)):
            values = scaled[i].tolist()
            values += values[:1]  # 闭合雷达图
            
            ax.plot(angles, values, linewidth=1.5, linestyle='solid', 
                    label=f"供应商 {top10.iloc[i]['供应商ID']}")
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), indicators)
        ax.set_rlabel_position(30)
        ax.set_ylim(0, 1)
        ax.set_title('Top10供应商指标雷达图', fontsize=14, pad=20)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig('Top10供应商指标雷达图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. 综合得分分布
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics['综合得分'], bins=20, kde=True, color='skyblue')
    if not top_50.empty:
        plt.axvline(top_50['综合得分'].min(), color='red', linestyle='--', 
                   label=f'Top50门槛值: {top_50["综合得分"].min():.3f}')
    plt.title('供应商综合得分分布')
    plt.xlabel('综合得分')
    plt.ylabel('供应商数量')
    plt.legend()
    plt.tight_layout()
    plt.savefig('供应商综合得分分布.png', dpi=300)
    plt.show()
    
    # 7. 各材料类型供应商得分比较
    if not top_50.empty and '材料分类' in top_50.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='材料分类', y='综合得分', data=top_50, showfliers=False)
        sns.stripplot(x='材料分类', y='综合得分', data=top_50, color='black', alpha=0.5, jitter=True)
        plt.title('不同材料类型供应商综合得分分布')
        plt.xlabel('材料类型')
        plt.ylabel('综合得分')
        plt.tight_layout()
        plt.savefig('不同材料类型供应商综合得分分布.png', dpi=300)
        plt.show()
    
    # 8. 各指标得分分布
    if not top_50.empty:
        plt.figure(figsize=(12, 8))
        indicators = ['CV', 'OFV', 'OFR', 'SRI', 'CA', 'PCI']
        melted = pd.melt(top_50, id_vars=['供应商ID'], value_vars=indicators,
                         var_name='指标', value_name='得分')
        
        plt.subplot(2, 1, 1)
        sns.boxplot(x='指标', y='得分', data=melted)
        plt.title('Top50供应商各指标得分分布')
        
        plt.subplot(2, 1, 2)
        sns.violinplot(x='指标', y='得分', data=melted, inner='quartile')
        plt.title('Top50供应商各指标得分密度分布')
        
        plt.tight_layout()
        plt.savefig('Top50供应商各指标得分密度分布.png', dpi=300)
        plt.show()

# 主函数
def main():
    # 加载并准备数据
    supply_data, order_data = load_and_prepare_data()
    
    # 预处理数据
    long_data = preprocess_data(supply_data, order_data)
    
    if long_data.empty:
        print("数据预处理后为空，请检查数据格式和列名")
        return pd.DataFrame()
    
    # 计算指标
    metrics = calculate_metrics(long_data)
    
    if metrics.empty:
        print("指标计算后为空，请检查数据")
        return pd.DataFrame()
    
    # 设置指标和权重
    indicators = ['CV', 'OFV', 'OFR', 'SRI', 'CA', 'PCI']
    weights = np.array([0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
    
    # TOPSIS评价
    metrics['综合得分'] = topsis_evaluation(metrics, weights, indicators)
    
    # 按综合得分排序
    metrics = metrics.sort_values('综合得分', ascending=False)
    
    # 选取Top50供应商
    top_50 = metrics.head(50).copy()
    
    # 单独绘制每个图表
    plot_separate_charts(metrics, top_50)
    
    # 保存结果
    result = top_50[['供应商ID', '材料分类', '综合得分']]
    result.to_csv('top_50_suppliers.csv', index=False)
    
    # 打印结果
    print("最重要的50家供应商：")
    print(result)
    
    # 材料分类统计
    material_counts = result['材料分类'].value_counts()
    print("\n供应商统计信息：")
    print(f"供应商总数: {len(metrics)}")
    print(f"Top50得分范围: {top_50['综合得分'].min():.4f} - {top_50['综合得分'].max():.4f}")
    print(f"材料分类分布: A类 {material_counts.get('A', 0)}家, B类 {material_counts.get('B', 0)}家, C类 {material_counts.get('C', 0)}家")
    
    return result

# 执行主函数
if __name__ == "__main__":
    top_suppliers = main()