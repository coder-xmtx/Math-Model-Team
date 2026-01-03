import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import random
from tqdm import tqdm
import os
from scipy import stats

# -------------------------- 基础设置 --------------------------
np.random.seed(42)
random.seed(42)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("figures", exist_ok=True)

# -------------------------- 离群点检测与去除函数 --------------------------
def remove_outliers(df, target_cols, method='IQR', z_threshold=3.0, log_info=True):
    df_cleaned = df.copy()
    original_rows = len(df_cleaned)
    
    for col in target_cols:
        if col not in df_cleaned.columns:
            if log_info:
                print(f"⚠️  列'{col}'不存在，跳过该列的离群点检测")
            continue
        
        df_cleaned = df_cleaned.dropna(subset=[col])
        if len(df_cleaned) == 0:
            if log_info:
                print(f"❌ 列'{col}'所有值为NaN，无法进行离群点检测")
            return df_cleaned
        
        if method == 'IQR':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
        elif method == 'Z-score':
            mean_val = df_cleaned[col].mean()
            std_val = df_cleaned[col].std()
            lower_bound = mean_val - z_threshold * std_val
            upper_bound = mean_val + z_threshold * std_val
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
        if log_info:
            removed = original_rows - len(df_cleaned)
            print(f"✅ 用{method}法去除'{col}'离群点：原始{original_rows}行 → 剩余{len(df_cleaned)}行（移除{removed}行）")
        original_rows = len(df_cleaned)
    
    return df_cleaned

# -------------------------- 数据预处理函数 --------------------------
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取数据，共{df.shape[0]}行，{df.shape[1]}列")
        return df
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

def filter_male_fetuses(df):
    if 'Y染色体浓度' not in df.columns:
        print("数据中不包含'Y染色体浓度'列")
        return None
    
    male_df = df[df['Y染色体浓度'] > 0].copy()
    print(f"筛选出男胎孕妇数据共{male_df.shape[0]}行")
    return male_df

def process_gestational_week(week_str):
    if pd.isna(week_str):
        return np.nan
    
    if 'w' in str(week_str) and '+' not in str(week_str):
        try:
            return float(re.findall(r'\d+\.?\d*', str(week_str))[0])
        except:
            return np.nan
    
    if 'w+' in str(week_str):
        parts = str(week_str).split('w+')
        if len(parts) == 2:
            try:
                return float(parts[0]) + float(parts[1])/7
            except:
                return np.nan
    
    try:
        return float(week_str)
    except:
        return np.nan

def process_detection_date(date_str):
    try:
        return datetime.strptime(str(date_str), '%Y%m%d')
    except:
        return np.nan

def process_duplicate_tests(df):
    agg_funcs = {
        '孕妇BMI': 'mean',
        'GC含量': 'mean',
        'Y染色体浓度': 'mean',
        '检测孕周': 'first',
        '检测日期': 'first',
        '怀孕次数': 'first',
        '生产次数': 'first',
        '年龄': 'first'
    }
    
    processed_df = df.groupby(['孕妇代码', '检测抽血次数']).agg(agg_funcs).reset_index()
    print(f"处理重复检验后的数据共{processed_df.shape[0]}行")
    return processed_df

def filter_gc_content(df):
    if 'GC含量' not in df.columns:
        print("数据中不包含'GC含量'列")
        return None
    
    filtered_df = df[(df['GC含量'] >= 0.4) & (df['GC含量'] <= 0.6)].copy()
    print(f"筛选GC含量合格的数据共{filtered_df.shape[0]}行")
    return filtered_df

def calculate_达标时间(df):
    grouped = df.groupby('孕妇代码')
    达标时间_dict = {}
    
    for 孕妇代码, group in grouped:
        sorted_group = group.sort_values('检测日期')
        达标记录 = sorted_group[sorted_group['Y染色体浓度'] >= 0.04]
        
        if not 达标记录.empty:
            first_record = sorted_group.iloc[0]
            达标时间_dict[孕妇代码] = {
                'BMI': group['孕妇BMI'].mean(),
                '达标时间': 达标记录.iloc[0]['检测孕周'],
                '怀孕次数': first_record['怀孕次数'],
                '生产次数': first_record['生产次数'],
                '年龄': first_record['年龄']
            }
    
    result_df = pd.DataFrame.from_dict(达标时间_dict, orient='index').reset_index()
    result_df.columns = ['孕妇代码', 'BMI', '达标时间', '怀孕次数', '生产次数', '年龄']
    print(f"成功计算{result_df.shape[0]}个孕妇的达标时间")
    return result_df

def preprocess_data(file_path, outlier_method='IQR', z_threshold=3.0):
    print("===== 数据预处理（含离群点去噪） =====")
    # 1. 读取数据
    df = load_data(file_path)
    if df is None:
        return None
    
    # 2. 筛选男胎
    male_df = filter_male_fetuses(df)
    if male_df is None:
        return None
    
    # 3. 处理孕周和日期
    male_df['检测孕周'] = male_df['检测孕周'].apply(process_gestational_week)
    male_df['检测日期'] = male_df['检测日期'].apply(process_detection_date)
    male_df = male_df.dropna(subset=['检测孕周', '检测日期'])
    print(f"处理孕周/日期后，剩余{male_df.shape[0]}行")
    
    # 4. 处理分类变量（怀孕次数/生产次数）
    for col in ['怀孕次数', '生产次数']:
        if col in male_df.columns and male_df[col].dtype == 'object':
            male_df[col] = male_df[col].apply(lambda x: re.findall(r'\d+', str(x))[0] if pd.notna(x) and re.findall(r'\d+', str(x)) else np.nan)
            male_df[col] = pd.to_numeric(male_df[col], errors='coerce')
    
    # 5. 剔除不合理记录（怀孕次数 < 生产次数）
    if '怀孕次数' in male_df.columns and '生产次数' in male_df.columns:
        male_df = male_df.dropna(subset=['怀孕次数', '生产次数'])
        invalid_mask = male_df['怀孕次数'] < male_df['生产次数']
        if invalid_mask.any():
            print(f"剔除{invalid_mask.sum()}条怀孕次数小于生产次数的不合理记录")
            male_df = male_df[~invalid_mask]
        print(f"剔除不合理记录后，剩余{male_df.shape[0]}行")
    
    # 6. 处理重复检验
    processed_df = process_duplicate_tests(male_df)
    
    # 7. 原始数据离群点去噪（BMI + Y染色体浓度）
    print("\n----- 原始数据离群点去噪（BMI + Y染色体浓度）-----")
    processed_df = remove_outliers(
        df=processed_df,
        target_cols=['孕妇BMI', 'Y染色体浓度'],
        method=outlier_method,
        z_threshold=z_threshold
    )
    if processed_df.empty:
        print("❌ 原始数据去噪后无剩余样本，终止预处理")
        return None
    
    # 8. 筛选GC含量
    gc_filtered_df = filter_gc_content(processed_df)
    if gc_filtered_df is None or gc_filtered_df.empty:
        print("❌ GC含量筛选后无剩余样本，终止预处理")
        return None
    
    # 9. 计算达标时间
    result_df = calculate_达标时间(gc_filtered_df)
    if result_df.empty:
        print("❌ 未计算出有效达标时间，终止预处理")
        return None
    
    # 10. 核心结果离群点去噪（BMI + 达标时间）
    print("\n----- 核心结果离群点去噪（BMI + 达标时间）-----")
    result_df = remove_outliers(
        df=result_df,
        target_cols=['BMI', '达标时间'],
        method=outlier_method,
        z_threshold=z_threshold
    )
    if result_df.empty:
        print("❌ 核心结果去噪后无剩余样本，终止预处理")
        return None
    
    print(f"\n预处理完成，最终有效样本数：{result_df.shape[0]}行")
    return result_df

# -------------------------- 相关性分析函数 --------------------------
def analyze_correlation(processed_df):
    print("\n" + "="*80)
    print("===== 各变量与Y染色体达标时间的相关性分析 =====")
    print("="*80)
    
    numeric_cols = ['BMI', '达标时间', '年龄', '怀孕次数', '生产次数']
    analysis_df = processed_df[numeric_cols].dropna()
    if len(analysis_df) < 3:
        print(f"❌ 有效样本量不足（仅{len(analysis_df)}个），无法进行相关性分析")
        return None
    
    # 计算相关系数矩阵
    corr_matrix = analysis_df.corr(method='pearson')
    print("\nPearson相关系数矩阵:")
    print(corr_matrix.round(4))
    
    # 单独分析每个变量与达标时间的相关性
    print("\n各变量与达标时间的相关性:")
    target_var = '达标时间'
    for col in analysis_df.columns:
        if col != target_var:
            pearson_r, pearson_p = stats.pearsonr(analysis_df[col], analysis_df[target_var])
            spearman_r, spearman_p = stats.spearmanr(analysis_df[col], analysis_df[target_var])
            
            print(f"\n- {col}:")
            print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.4f}")
            print(f"  Spearman r = {spearman_r:.4f}, p = {spearman_p:.4f}")
    
    return corr_matrix

# -------------------------- 决策树分箱核心函数（新增/重构） --------------------------
def extract_feature_bins_from_tree(model, feature_names, df):
    """
    从决策树中提取**特征分箱边界**，生成结构化分箱表
    返回：分箱列表（每个分箱含边界、样本筛选逻辑）
    """
    if model is None:
        print("❌ 无有效决策树模型，无法提取分箱")
        return []
    
    # 决策树核心信息
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left  # 左子节点（<=阈值）
    children_right = model.tree_.children_right  # 右子节点（>阈值）
    feature_idx = model.tree_.feature  # 每个节点分裂的特征索引
    thresholds = model.tree_.threshold  # 每个节点的分裂阈值
    feature_names = feature_names  # 特征名称映射
    
    # 遍历所有叶节点，提取分箱规则（路径回溯）
    bins = []
    stack = [(0, {})]  # (当前节点ID, 该节点的分箱规则：{特征名: (下界, 上界)})
    
    while stack:
        node_id, current_rules = stack.pop()
        
        # 若为叶节点：保存当前分箱规则
        if children_left[node_id] == children_right[node_id]:
            # 生成分箱描述（如：BMI∈(18.5,24.0] & 年龄∈(25,35]）
            bin_desc_parts = []
            for feat, (lower, upper) in current_rules.items():
                if lower == -np.inf:
                    part = f"{feat}≤{upper:.1f}"
                elif upper == np.inf:
                    part = f"{feat}>{lower:.1f}"
                else:
                    part = f"{feat}∈({lower:.1f},{upper:.1f}]"
                bin_desc_parts.append(part)
            bin_desc = " & ".join(bin_desc_parts) if bin_desc_parts else "全样本"
            
            # 计算该分箱的样本量和达标时间统计
            sample_mask = np.ones(len(df), dtype=bool)
            for feat, (lower, upper) in current_rules.items():
                if lower != -np.inf:
                    sample_mask &= (df[feat] > lower)
                if upper != np.inf:
                    sample_mask &= (df[feat] <= upper)
            bin_samples = df[sample_mask]
            
            if len(bin_samples) < 5:  # 过滤样本过少的分箱（避免不稳定）
                continue
            
            # 达标时间统计
            time_mean = bin_samples['达标时间'].mean()
            time_median = bin_samples['达标时间'].median()
            time_95p = np.percentile(bin_samples['达标时间'], 95)  # 最佳检测时点（95%分位数）
            
            bins.append({
                '分箱编号': len(bins) + 1,
                '分箱描述': bin_desc,
                '分箱规则': current_rules,  # 用于后续筛选样本
                '样本数': len(bin_samples),
                '达标时间均值(周)': round(time_mean, 2),
                '达标时间中位数(周)': round(time_median, 2),
                '最佳检测时点(95%分位数,周)': round(time_95p, 2)
            })
        
        # 若为非叶节点：递归处理子节点，更新分箱规则
        else:
            split_feat = feature_names[feature_idx[node_id]]
            split_thr = thresholds[node_id]
            
            # 1. 处理右子节点（>阈值）：更新该特征的下界
            right_rules = current_rules.copy()
            if split_feat not in right_rules:
                right_rules[split_feat] = (-np.inf, np.inf)
            # 右子节点：特征>阈值 → 下界更新为split_thr
            right_rules[split_feat] = (split_thr, right_rules[split_feat][1])
            stack.append((children_right[node_id], right_rules))
            
            # 2. 处理左子节点（<=阈值）：更新该特征的上界
            left_rules = current_rules.copy()
            if split_feat not in left_rules:
                left_rules[split_feat] = (-np.inf, np.inf)
            # 左子节点：特征<=阈值 → 上界更新为split_thr
            left_rules[split_feat] = (left_rules[split_feat][0], split_thr)
            stack.append((children_left[node_id], left_rules))
    
    # 转换为DataFrame便于查看
    bins_df = pd.DataFrame(bins) if bins else pd.DataFrame()
    return bins_df

def generate_binned_optimal_time(bins_df):
    """
    从分箱表中提取「分箱-最佳时点」映射，用于可视化和应用
    """
    if bins_df.empty:
        return pd.DataFrame()
    
    optimal_time_df = bins_df[['分箱编号', '分箱描述', '样本数', '最佳检测时点(95%分位数,周)']].copy()
    optimal_time_df.columns = ['分组ID', '分组描述', '样本数', '最佳NIPT时点(95th)']
    return optimal_time_df

# -------------------------- 决策树建模（适配分箱逻辑） --------------------------
def build_decision_tree_for_binning(df, max_depth=3, min_samples_leaf=0.1):
    """
    构建用于分箱的决策树（控制深度和最小样本数，确保分箱合理性）
    """
    print("\n===== 构建决策树（用于特征分箱） =====")
    
    # 特征选择（可调整，此处用BMI、年龄、怀孕次数、生产次数）
    feature_cols = ['BMI', '年龄', '怀孕次数', '生产次数']
    X = df[feature_cols].dropna()
    y = df.loc[X.index, '达标时间']  # 确保X和y索引对齐
    
    # 样本量校验
    if len(X) < 20:  # 至少20个样本才建模（避免分箱不稳定）
        print(f"❌ 样本量不足（仅{len(X)}行），无法构建分箱决策树")
        return None, X, y, feature_cols
    
    # 决策树参数：控制分箱粒度（max_depth越小，分箱越粗；min_samples_leaf越大，分箱越稳定）
    min_samples = max(int(min_samples_leaf * len(X)), 5)  # 每个叶节点至少5个样本
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples,
        random_state=42,
        criterion='squared_error'  # 回归树标准（最小化MSE）
    )
    model.fit(X, y)
    
    # 模型评估（交叉验证R²，反映模型解释力）
    cv_folds = min(5, len(X)//5)  # 避免折叠数过多
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    print(f"决策树交叉验证R²得分: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 特征重要性（判断哪些特征对分箱贡献大）
    print("\n特征重要性（用于分箱的关键特征）:")
    for feat, imp in zip(feature_cols, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    # 打印决策树规则（便于理解分箱逻辑）
    tree_rules = export_text(model, feature_names=feature_cols)
    print("\n决策树分箱规则:")
    print(tree_rules)
    
    return model, X, y, feature_cols

# -------------------------- 可视化函数（适配分箱结果） --------------------------
def plot_feature_importance(feature_importance, feature_names):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)[::-1]
    
    plt.bar(range(len(feature_importance)), feature_importance[indices], align='center', color='#2E86AB')
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title('特征重要性排序（分箱决策树）', fontsize=14, fontweight='bold')
    plt.ylabel('特征重要性', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 特征重要性图已保存至 figures/feature_importance.png")

def plot_bmi_time_scatter(df, correlation_results=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='BMI', y='达标时间', data=df, alpha=0.6, s=60, color='#2E86AB')
    
    # 趋势线
    z = np.polyfit(df['BMI'], df['达标时间'], 1)
    p = np.poly1d(z)
    plt.plot(df['BMI'], p(df['BMI']), "r--", linewidth=2, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # 相关性标注
    if correlation_results is not None and '达标时间' in correlation_results.columns:
        pearson_r = correlation_results.loc['BMI', '达标时间']
        plt.text(
            0.05, 0.95, 
            f"Pearson r = {pearson_r:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            fontsize=11,
            verticalalignment='top'
        )
    
    plt.title('孕妇BMI与NIPT达标时间关系', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('孕妇BMI (kg/m²)', fontsize=12)
    plt.ylabel('NIPT达标时间（孕周）', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/bmi_time_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ BMI-达标时间散点图已保存至 figures/bmi_time_scatter.png")

def plot_decision_tree_structure(model, feature_names):
    plt.figure(figsize=(20, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True,
        node_ids=True,
        impurity=True,
        class_names=None  # 回归树无类别名
    )
    plt.title('决策树结构（用于NIPT达标时间分箱）', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 决策树结构图已保存至 figures/decision_tree.png")

def plot_binned_optimal_time(optimal_time_df):
    """可视化分箱后的最佳检测时点"""
    if optimal_time_df.empty:
        print("❌ 无分箱数据，无法绘制最佳时点图")
        return
    
    plt.figure(figsize=(14, 8))
    # 按分箱编号排序（确保逻辑顺序）
    optimal_time_df = optimal_time_df.sort_values('分组ID')
    
    # 绘制柱状图
    bars = plt.bar(
        x=range(len(optimal_time_df)),
        height=optimal_time_df['最佳NIPT时点(95th)'],
        color=plt.cm.Blues(np.linspace(0.4, 0.8, len(optimal_time_df))),
        edgecolor='white',
        linewidth=1.5
    )
    
    # 标注：分箱描述、样本数、最佳时点
    for i, (_, row) in enumerate(optimal_time_df.iterrows()):
        bar = bars[i]
        # 分箱描述换行（避免过长）
        desc = row['分组描述'].replace(' & ', '\n')
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{desc}\n最佳时点: {row['最佳NIPT时点(95th)']:.2f}周\n(n={row['样本数']})",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    
    plt.title('决策树分箱后各分组的最佳NIPT检测时点（95%分位数）', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('分箱分组', fontsize=12)
    plt.ylabel('最佳NIPT检测时点（孕周）', fontsize=12)
    plt.xticks(range(len(optimal_time_df)), [f"分组{row['分组ID']}" for _, row in optimal_time_df.iterrows()])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/binned_optimal_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 分箱最佳时点图已保存至 figures/binned_optimal_time.png")

# -------------------------- 蒙特卡洛模拟函数（适配分箱逻辑） --------------------------
def simulate_measurement_error(original_df, raw_data, outlier_method='IQR', z_threshold=3.0, sigma=0.005, n_simulations=1000):
    print("\n===== 开始蒙特卡洛模拟（分析检测误差对分箱结果的影响） =====")
    
    required_cols = ['孕妇代码', '检测抽血次数', '检测日期', '检测孕周', '孕妇BMI', 'GC含量', 'Y染色体浓度', '怀孕次数', '生产次数', '年龄']
    if not all(col in raw_data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in raw_data.columns]
        print(f"❌ 原始数据缺少必要列：{missing_cols}")
        return [], pd.DataFrame()
    
    # 预处理原始数据（与主流程一致）
    simulation_data = raw_data[required_cols].copy()
    simulation_data['检测孕周'] = simulation_data['检测孕周'].apply(process_gestational_week)
    simulation_data['检测日期'] = simulation_data['检测日期'].apply(process_detection_date)
    
    # 处理分类变量
    for col in ['怀孕次数', '生产次数']:
        if col in simulation_data.columns and simulation_data[col].dtype == 'object':
            simulation_data[col] = simulation_data[col].apply(lambda x: re.findall(r'\d+', str(x))[0] if pd.notna(x) and re.findall(r'\d+', str(x)) else np.nan)
            simulation_data[col] = pd.to_numeric(simulation_data[col], errors='coerce')
    
    # 剔除不合理记录
    simulation_data = simulation_data.dropna(subset=['怀孕次数', '生产次数'])
    simulation_data = simulation_data[simulation_data['怀孕次数'] >= simulation_data['生产次数']]
    simulation_data = simulation_data[(simulation_data['GC含量'] >= 0.4) & (simulation_data['GC含量'] <= 0.6)]
    simulation_data = simulation_data.dropna(subset=['检测孕周', '检测日期'])
    
    # 按孕妇分组
    孕妇_groups = simulation_data.groupby('孕妇代码')
    all_binned_timepoints = []
    
    # 模拟循环
    for sim in tqdm(range(n_simulations), desc="蒙特卡洛模拟进度"):
        temp_results = []
        
        for 孕妇代码, group in 孕妇_groups:
            # 添加检测误差（模拟Y染色体浓度波动）
            group_with_error = group.copy()
            group_with_error['Y染色体浓度_error'] = group_with_error['Y染色体浓度'] + np.random.normal(0, sigma, len(group_with_error))
            
            # 处理重复检验
            processed_group = group_with_error.groupby('检测抽血次数').agg({
                '孕妇BMI': 'mean',
                '检测孕周': 'first',
                '检测日期': 'first',
                'Y染色体浓度_error': 'mean',
                '怀孕次数': 'first',
                '生产次数': 'first',
                '年龄': 'first'
            }).reset_index()
            
            # 计算达标时间
            processed_group = processed_group.sort_values('检测日期')
            达标记录 = processed_group[processed_group['Y染色体浓度_error'] >= 0.04]
            if not 达标记录.empty:
                达标时间 = 达标记录.iloc[0]['检测孕周']
                if not pd.isna(达标时间):
                    temp_results.append({
                        '孕妇代码': 孕妇代码,
                        'BMI': processed_group['孕妇BMI'].mean(),
                        '达标时间': 达标时间,
                        '怀孕次数': processed_group['怀孕次数'].iloc[0],
                        '生产次数': processed_group['生产次数'].iloc[0],
                        '年龄': processed_group['年龄'].iloc[0]
                    })
        
        # 过滤无效样本并去噪
        sim_df = pd.DataFrame(temp_results).dropna(subset=['BMI', '达标时间', '怀孕次数', '生产次数', '年龄'])
        if len(sim_df) < 20:
            continue
        sim_df = remove_outliers(sim_df, ['BMI', '达标时间'], method=outlier_method, log_info=False)
        if sim_df.empty:
            continue
        
        # 构建临时决策树并分箱
        model_sim, X_sim, y_sim, feat_cols = build_decision_tree_for_binning(sim_df, max_depth=3, min_samples_leaf=0.1)
        if model_sim is None:
            continue
        
        # 提取分箱并记录最佳时点
        bins_df_sim = extract_feature_bins_from_tree(model_sim, feat_cols, sim_df)
        if not bins_df_sim.empty:
            optimal_time_sim = generate_binned_optimal_time(bins_df_sim)
            for _, row in optimal_time_sim.iterrows():
                all_binned_timepoints.append({
                    '模拟次数': sim,
                    '分组ID': row['分组ID'],
                    '分组描述': row['分组描述'],
                    '最佳NIPT时点': row['最佳NIPT时点(95th)'],
                    '样本数': row['样本数']
                })
    
    # 保存模拟结果
    timepoints_df = pd.DataFrame(all_binned_timepoints)
    if not timepoints_df.empty:
        timepoints_df.to_csv("simulation_binned_timepoints.csv", index=False, encoding='utf-8-sig')
        print("✅ 分箱模拟时点结果已保存至 simulation_binned_timepoints.csv")
        
        # 分箱稳定性统计（变异系数：越小越稳定）
        group_stats = timepoints_df.groupby('分组描述')['最佳NIPT时点'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        group_stats['变异系数(%)'] = (group_stats['std'] / group_stats['mean'] * 100).round(2)
        group_stats.columns = ['有效模拟次数', '最佳时点均值(周)', '最佳时点标准差(周)', '最佳时点最小值(周)', '最佳时点最大值(周)', '变异系数(%)']
        group_stats.to_csv("monte_carlo_binned_stats.csv", encoding='utf-8-sig')
        
        # 打印统计结果
        print("\n" + "="*80)
        print("蒙特卡洛模拟：分箱最佳NIPT时点稳定性统计（变异系数越小越稳定）")
        print("="*80)
        print(group_stats.to_string())
        
        # 可视化模拟结果
        plot_monte_carlo_binned_results(timepoints_df, group_stats)
    else:
        print("❌ 蒙特卡洛模拟无有效分箱结果")
    
    return all_binned_timepoints, timepoints_df

def plot_monte_carlo_binned_results(timepoints_df, group_stats):
    """可视化分箱模拟结果的稳定性"""
    # 1. 各分箱最佳时点分布箱线图
    plt.figure(figsize=(16, 8))
    # 按变异系数排序（不稳定的分箱放前面）
    group_order = group_stats.sort_values('变异系数(%)', ascending=False).index
    sns.boxplot(x='分组描述', y='最佳NIPT时点', data=timepoints_df, order=group_order, palette='viridis')
    
    # 添加均值点
    means = timepoints_df.groupby('分组描述')['最佳NIPT时点'].mean()
    positions = range(len(group_order))
    plt.scatter(positions, [means[g] for g in group_order], color='red', marker='D', s=60, label='均值')
    
    plt.title('蒙特卡洛模拟：各分箱最佳NIPT时点分布（检测误差影响）', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('分箱分组', fontsize=12)
    plt.ylabel('最佳NIPT时点（孕周）', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/monte_carlo_binned_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 分箱变异系数柱状图
    plt.figure(figsize=(14, 6))
    group_stats_sorted = group_stats.sort_values('变异系数(%)', ascending=False)
    bars = plt.bar(
        x=range(len(group_stats_sorted)),
        height=group_stats_sorted['变异系数(%)'],
        color=plt.cm.Reds(np.linspace(0.4, 0.8, len(group_stats_sorted))),
        edgecolor='white',
        linewidth=1.5
    )
    
    # 标注变异系数
    for i, (idx, row) in enumerate(group_stats_sorted.iterrows()):
        bar = bars[i]
        # 分箱描述简化（取前30字符）
        short_desc = idx[:30] + "..." if len(idx) > 30 else idx
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{short_desc}\n{row['变异系数(%)']:.1f}%",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    
    plt.title('各分箱最佳NIPT时点的变异系数（检测误差敏感性）', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('分箱分组', fontsize=12)
    plt.ylabel('变异系数(%)', fontsize=12)
    plt.xticks(range(len(group_stats_sorted)), [f"分组{i+1}" for i in range(len(group_stats_sorted))])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/monte_carlo_binned_cv.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 蒙特卡洛分箱模拟图已保存至 figures/ 目录")

# -------------------------- 主函数（分箱逻辑整合） --------------------------
def main(
    file_path="data/data-total.xlsx",
    max_depth=3,  # 决策树深度（控制分箱数量，3→约4-8个分箱）
    min_samples_leaf=0.1,  # 叶节点最小样本比例（避免样本过少的分箱）
    outlier_method='IQR',
    z_threshold=3.0,
    sigma=0.03,  # 检测误差标准差
    n_simulations=1000  # 蒙特卡洛模拟次数
):
    """主函数：完整分箱分析流程"""
    # 1. 数据预处理
    processed_df = preprocess_data(
        file_path=file_path,
        outlier_method=outlier_method,
        z_threshold=z_threshold
    )
    if processed_df is None or processed_df.empty:
        print("数据预处理失败，无法继续分析")
        return None, None, None, None
    
    # 保存预处理数据
    processed_df.to_csv("processed_data_denoised.csv", index=False, encoding='utf-8-sig')
    print("✅ 预处理数据已保存为 processed_data_denoised.csv")
    
    # 2. 相关性分析
    correlation_results = analyze_correlation(processed_df)
    
    # 3. 可视化：BMI-达标时间散点图
    plot_bmi_time_scatter(processed_df, correlation_results)
    
    # 4. 构建决策树（用于分箱）
    model, X, y, feature_names = build_decision_tree_for_binning(
        df=processed_df,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )
    if model is None:
        print("决策树建模失败，后续分析终止")
        return processed_df, None, None, correlation_results
    
    # 5. 可视化：特征重要性、决策树结构
    plot_feature_importance(model.feature_importances_, feature_names)
    plot_decision_tree_structure(model, feature_names)
    
    # 6. 决策树分箱：提取分箱规则并计算最佳时点
    print("\n===== 决策树分箱结果 =====")
    bins_df = extract_feature_bins_from_tree(model, feature_names, processed_df)
    if bins_df.empty:
        print("❌ 未提取到有效分箱")
        return processed_df, model, None, correlation_results
    
    # 保存分箱详细结果（含规则和统计）
    bins_df.to_csv("decision_tree_bins_detail.csv", index=False, encoding='utf-8-sig')
    print("✅ 分箱详细结果已保存为 decision_tree_bins_detail.csv")
    print("\n分箱详细信息:")
    print(bins_df[['分箱编号', '分箱描述', '样本数', '最佳检测时点(95%分位数,周)']].to_string(index=False))
    
    # 提取最佳时点用于可视化
    optimal_time_df = generate_binned_optimal_time(bins_df)
    optimal_time_df.to_csv("binned_optimal_nipt_time.csv", index=False, encoding='utf-8-sig')
    print("✅ 分箱最佳时点已保存为 binned_optimal_nipt_time.csv")
    
    # 7. 可视化：分箱最佳时点
    plot_binned_optimal_time(optimal_time_df)
    
    # 8. 蒙特卡洛模拟（分析检测误差对分箱的影响）
    raw_data = load_data(file_path)
    sim_results, sim_timepoints_df = [], pd.DataFrame()
    if raw_data is not None:
        raw_male_data = filter_male_fetuses(raw_data)
        if raw_male_data is not None:
            sim_results, sim_timepoints_df = simulate_measurement_error(
                original_df=processed_df,
                raw_data=raw_male_data,
                outlier_method=outlier_method,
                z_threshold=z_threshold,
                sigma=sigma,
                n_simulations=n_simulations
            )
    
    # 输出核心结论
    print("\n" + "="*80)
    print("===== 核心分箱结论 =====")
    print("="*80)
    print("1. 关键分箱特征（按重要性排序）:")
    for feat, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"   - {feat}: 重要性 {imp:.4f}")
    
    print("\n2. 各分箱最佳NIPT检测时点:")
    for _, row in optimal_time_df.iterrows():
        print(f"   - {row['分组描述']}: {row['最佳NIPT时点(95th)']:.2f}周（样本数：{row['样本数']}）")
    
    if not sim_timepoints_df.empty:
        print("\n3. 检测误差影响：")
        stable_groups = sim_timepoints_df.groupby('分组描述')['最佳NIPT时点'].agg('std').sort_values().head(2)
        print(f"   - 最稳定的2个分箱（标准差最小）:")
        for group, std in stable_groups.items():
            print(f"     * {group[:40]}...: 标准差 {std:.4f}周")
    
    return processed_df, model, bins_df, correlation_results

# -------------------------- 执行分箱分析 --------------------------
if __name__ == "__main__":
    processed_df, model, bins_df, correlation_results = main(
        file_path="data/data-total.xlsx",  # 请替换为你的数据路径
        max_depth=3,
        min_samples_leaf=0.1,
        outlier_method='IQR',
        sigma=0.03,
        n_simulations=1000
    )