import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
print("1. 数据加载与预处理...")
df = pd.read_excel('data/data-total.xlsx')

# 转换检测日期格式（如20230409 -> 2023-04-09）
def convert_date(date_val):
    if pd.isna(date_val):
        return pd.NaT
    try:
        date_str = str(int(date_val))
        if len(date_str) == 8:
            return pd.to_datetime(date_str, format='%Y%m%d')
        else:
            return pd.NaT
    except:
        return pd.NaT

df['检测日期'] = df['检测日期'].apply(convert_date)

# 转换检测孕周为周数（小数表示）
def convert_gestational_week(week_str):
    if pd.isna(week_str):
        return np.nan
    if isinstance(week_str, (int, float)):
        return float(week_str)
    
    week_str = str(week_str).replace('w', '').replace('W', '').strip()
    
    if '+' in week_str:
        parts = week_str.split('+')
        week = float(parts[0])
        day = float(parts[1]) if len(parts) > 1 else 0
        return week + day/7.0
    else:
        try:
            return float(week_str)
        except:
            return np.nan

df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)

# 处理GC含量过滤
df = df[(df['GC含量'] >= 0.4) & (df['GC含量'] <= 0.6)]

# 处理同一管血多次检测的情况
df_grouped = df.groupby(['孕妇代码', '检测日期']).agg({
    'Y染色体浓度': 'mean',
    'GC含量': 'mean',
    '孕妇BMI': 'first',
    '检测孕周数值': 'first',
    '年龄': 'first',
    '体重': 'first',
    '身高': 'first'
}).reset_index()

# 按检测日期排序
df_grouped = df_grouped.sort_values(['孕妇代码', '检测日期'])

# 计算每个孕妇的达标时间（首次达到4%的孕周）
df_grouped['达标否'] = (df_grouped['Y染色体浓度'] >= 0.04).astype(int)

# 找到每个孕妇首次达标的时间
first_reach = df_grouped[df_grouped['达标否'] == 1].groupby('孕妇代码').first().reset_index()
first_reach = first_reach[['孕妇代码', '检测孕周数值']].rename(columns={'检测孕周数值': '达标孕周'})

# 合并回原数据集
df_merged = pd.merge(df_grouped, first_reach, on='孕妇代码', how='left')
df_merged['是否首次达标'] = (df_merged['检测孕周数值'] == df_merged['达标孕周']).astype(int)

# 过滤掉从未达标的孕妇
valid_women = df_merged.groupby('孕妇代码')['达标否'].max().reset_index()
valid_women = valid_women[valid_women['达标否'] == 1]['孕妇代码']
df_final = df_merged[df_merged['孕妇代码'].isin(valid_women)]

print(f"处理后数据集形状: {df_final.shape}")

# 2. 特征工程
print("\n2. 特征工程...")
# 选择特征
features = ['孕妇BMI', '检测孕周数值', '年龄', '体重', '身高']
feature_names = features  # 保存特征名称用于规则提取
X = df_final[features]
y = df_final['达标否']

# 3. 第一阶段：基于决策树的风险分组
print("\n3. 第一阶段：基于决策树的风险分组...")
# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 调整决策树参数，避免过拟合
param_grid = {
    'max_depth': [3, 4],
    'min_samples_split': [20, 30],
    'min_samples_leaf': [10, 15]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数训练决策树
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 提取决策树分组规则
df_final['Group_Label'] = best_dt.apply(X)

# 获取所有叶节点
n_nodes = best_dt.tree_.node_count
children_left = best_dt.tree_.children_left
children_right = best_dt.tree_.children_right
feature = best_dt.tree_.feature
threshold = best_dt.tree_.threshold

# 找到所有叶节点
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # 初始节点
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    is_leaves[node_id] = (children_left[node_id] == children_right[node_id])

    if not is_leaves[node_id]:
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))

leaf_nodes = np.where(is_leaves)[0]
print(f"叶节点数量: {len(leaf_nodes)}")

# 提取每个叶节点的规则
def extract_rules_from_leaf(node_id, tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    
    rule_parts = []
    while node_id != 0:
        parent = np.where((left == node_id) | (right == node_id))[0][0]
        if left[parent] == node_id:
            rule_parts.append(f"{features[parent]} <= {threshold[parent]:.2f}")
        else:
            rule_parts.append(f"{features[parent]} > {threshold[parent]:.2f}")
        node_id = parent
    
    return " & ".join(reversed(rule_parts))

# 创建分组规则映射
group_rules_map = {}
for node_id in leaf_nodes:
    rule = extract_rules_from_leaf(node_id, best_dt, feature_names)
    group_rules_map[node_id] = rule
    print(f"叶节点 {node_id}: {rule}")

# 4. 第二阶段：基于随机森林的组内风险预测与优化
print("\n4. 第二阶段：基于随机森林的组内风险预测与优化...")
# 获取唯一分组标签
unique_groups = np.unique(df_final['Group_Label'])

# 为每个分组建立随机森林模型
group_models = {}
group_results = {}

for group in unique_groups:
    print(f"\n处理分组 {group}...")
    group_data = df_final[df_final['Group_Label'] == group]
    
    # 检查样本量
    if len(group_data) < 20:  # 提高最小样本量要求
        print(f"分组 {group} 数据量不足 ({len(group_data)} 样本)，跳过")
        continue
    
    # 组内特征（排除BMI）
    group_features = ['检测孕周数值', '年龄', '体重', '身高']
    X_group = group_data[group_features]
    y_group = group_data['达标否']
    
    # 检查目标变量的类别分布
    unique_classes = np.unique(y_group)
    if len(unique_classes) == 1:
        print(f"分组 {group} 只有一个类别 ({unique_classes[0]})，跳过模型训练")
        # 如果只有一类，直接使用该类别的概率
        if unique_classes[0] == 1:  # 全部达标
            proba_all = 1.0
        else:  # 全部未达标
            proba_all = 0.0
    else:
        # 训练随机森林模型
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_group, y_group)
        group_models[group] = rf
    
    # 生成失败风险曲线
    # 固定其他特征取中位数，变化孕周
    other_features_median = X_group.drop('检测孕周数值', axis=1).median()
    
    # 创建模拟数据
    min_week = X_group['检测孕周数值'].min()
    max_week = X_group['检测孕周数值'].max()
    weeks = np.linspace(min_week, max_week, 100)
    
    # 准备预测数据
    X_pred = pd.DataFrame({
        '检测孕周数值': weeks
    })
    for col, val in other_features_median.items():
        X_pred[col] = val
    
    # 预测达标概率
    if len(unique_classes) == 1:
        # 只有一个类别的情况
        proba = np.full(len(weeks), proba_all)
    else:
        # 正常情况，使用模型预测
        proba_result = rf.predict_proba(X_pred)
        # 检查概率数组的形状
        if proba_result.shape[1] == 1:
            # 如果只有一列，说明模型只预测了一个类别
            if rf.classes_[0] == 1:  # 模型只预测了达标
                proba = np.ones(len(weeks))
            else:  # 模型只预测了未达标
                proba = np.zeros(len(weeks))
        else:
            # 正常情况，有两列概率
            proba = proba_result[:, 1]
    
    failure_risk = 1 - proba
    
    # 找到最佳时点 t* (达标概率 >= 0.95 的最小孕周)
    valid_weeks = weeks[proba >= 0.95]
    if len(valid_weeks) > 0:
        t_star = valid_weeks[0]
    else:
        t_star = max_week  # 如果没有达标周数，取最大值
    
    group_results[group] = {
        'weeks': weeks,
        'proba': proba,
        'failure_risk': failure_risk,
        't_star': t_star
    }
    
    print(f"分组 {group} 的最佳时点: {t_star:.2f} 周")

# 5. 第三阶段：蒙特卡洛模拟验证
print("\n5. 第三阶段：蒙特卡洛模拟验证...")
n_simulations = 50  # 减少模拟次数以提高速度
sigma = 0.1  # 噪声标准差

# 存储每次模拟的结果
simulation_results = {group: [] for group in unique_groups}

for i in range(n_simulations):
    print(f"进行第 {i+1}/{n_simulations} 次模拟...")
    
    # 添加噪声到关键特征
    df_noisy = df_final.copy()
    noisy_features = ['检测孕周数值', '孕妇BMI']
    
    for feature in noisy_features:
        noise = np.random.normal(0, sigma, len(df_noisy))
        df_noisy[feature] = df_noisy[feature] + noise
    
    # 使用决策树重新分组
    X_noisy = df_noisy[features]
    df_noisy['Group_Label_Noisy'] = best_dt.apply(X_noisy)
    
    # 对每个分组重新计算最佳时点
    for group in unique_groups:
        if group not in group_results:  # 只处理有结果的分组
            continue
            
        group_data = df_noisy[df_noisy['Group_Label_Noisy'] == group]
        
        if len(group_data) < 20:  # 数据量太少，跳过
            continue
        
        # 组内特征（排除BMI）
        group_features = ['检测孕周数值', '年龄', '体重', '身高']
        X_group = group_data[group_features]
        y_group = group_data['达标否']
        
        # 检查目标变量的类别分布
        unique_classes = np.unique(y_group)
        if len(unique_classes) == 1:
            # 如果只有一类，直接使用该类别的概率
            if unique_classes[0] == 1:  # 全部达标
                proba_all = 1.0
            else:  # 全部未达标
                proba_all = 0.0
            
            # 创建模拟数据
            min_week = X_group['检测孕周数值'].min()
            max_week = X_group['检测孕周数值'].max()
            weeks = np.linspace(min_week, max_week, 50)
            
            # 预测达标概率
            proba = np.full(len(weeks), proba_all)
        else:
            # 训练随机森林模型
            rf = RandomForestClassifier(n_estimators=50, random_state=42+i)
            rf.fit(X_group, y_group)
            
            # 生成失败风险曲线
            other_features_median = X_group.drop('检测孕周数值', axis=1).median()
            
            min_week = X_group['检测孕周数值'].min()
            max_week = X_group['检测孕周数值'].max()
            weeks = np.linspace(min_week, max_week, 50)
            
            X_pred = pd.DataFrame({
                '检测孕周数值': weeks
            })
            for col, val in other_features_median.items():
                X_pred[col] = val
            
            # 预测达标概率
            proba_result = rf.predict_proba(X_pred)
            # 检查概率数组的形状
            if proba_result.shape[1] == 1:
                # 如果只有一列，说明模型只预测了一个类别
                if rf.classes_[0] == 1:  # 模型只预测了达标
                    proba = np.ones(len(weeks))
                else:  # 模型只预测了未达标
                    proba = np.zeros(len(weeks))
            else:
                # 正常情况，有两列概率
                proba = proba_result[:, 1]
        
        # 找到最佳时点 t*
        valid_weeks = weeks[proba >= 0.95]
        if len(valid_weeks) > 0:
            t_star = valid_weeks[0]
            simulation_results[group].append(t_star)

# 分析模拟结果
final_recommendations = []

for group in unique_groups:
    if group not in simulation_results or len(simulation_results[group]) == 0:
        continue
    
    t_values = np.array(simulation_results[group])
    mean_t = np.mean(t_values)
    std_t = np.std(t_values)
    ci_lower = np.percentile(t_values, 2.5)
    ci_upper = np.percentile(t_values, 97.5)
    
    # 决定最终推荐时点
    if std_t < 0.2:
        final_t = mean_t
    else:
        final_t = ci_upper  # 使用上置信区间作为推荐
    
    # 获取分组描述
    group_desc = f"Group {group}"
    
    # 获取理论最佳时点
    theoretical_t = group_results[group]['t_star'] if group in group_results else mean_t
    
    # 获取分组规则
    group_rule = group_rules_map.get(group, "规则未找到")
    
    final_recommendations.append({
        '分组': group_desc,
        '分组特征描述': group_rule,
        '理论最佳时点': round(theoretical_t, 2),
        '稳健推荐时点': round(final_t, 2),
        '预期达标率': 0.95  # 我们的目标达标率
    })

# 6. 结果输出与可视化
print("\n6. 结果输出与可视化...")

# 创建结果表格
results_df = pd.DataFrame(final_recommendations)
print("\n最佳时点推荐表:")
print(results_df.to_string(index=False))

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(best_dt, feature_names=features, filled=True, rounded=True, class_names=['未达标', '达标'])
plt.title("决策树分组模型")
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# 可视化每个分组的失败风险曲线
for group in group_results:
    weeks = group_results[group]['weeks']
    failure_risk = group_results[group]['failure_risk']
    t_star = group_results[group]['t_star']
    
    plt.figure(figsize=(10, 6))
    plt.plot(weeks, failure_risk, 'b-', label='失败风险')
    plt.axvline(x=t_star, color='r', linestyle='--', label=f'最佳时点: {t_star:.2f}周')
    plt.axhline(y=0.05, color='g', linestyle='--', label='可接受风险阈值(5%)')
    plt.xlabel('检测孕周')
    plt.ylabel('失败风险')
    plt.title(f'分组 {group} 的失败风险曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'failure_risk_group_{group}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 可视化每个分组的最佳时点分布
plt.figure(figsize=(12, 8))
for group in simulation_results:
    if len(simulation_results[group]) > 0:
        plt.boxplot(simulation_results[group], positions=[group], widths=0.6)
        plt.text(group, np.median(simulation_results[group]), f'{np.median(simulation_results[group]):.2f}', 
                ha='center', va='bottom', fontweight='bold')

plt.xlabel('分组')
plt.ylabel('最佳时点(周)')
plt.title('各分组最佳时点分布(蒙特卡洛模拟)')
plt.xticks(unique_groups, [f'Group {g}' for g in unique_groups])
plt.grid(True, axis='y')
plt.savefig('t_star_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存详细的分组规则到文件
with open('group_rules.txt', 'w', encoding='utf-8') as f:
    f.write("决策树分组规则详情:\n")
    f.write("="*50 + "\n")
    for group, rule in group_rules_map.items():
        group_data = df_final[df_final['Group_Label'] == group]
        if len(group_data) > 0:
            f.write(f"Group {group}:\n")
            f.write(f"  规则: {rule}\n")
            f.write(f"  样本数: {len(group_data)}\n")
            f.write(f"  达标率: {group_data['达标否'].mean():.2f}\n")
            f.write("-"*50 + "\n")

print("\n分析完成! 结果已保存到当前目录下的图像文件和文本文件中。")