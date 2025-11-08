import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from tqdm import tqdm
import pickle
import os
from scipy import stats
from scipy.signal import savgol_filter

# 设置中文显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 数据预处理函数
def convert_gestational_week(week_str):
    """将孕周字符串转换为数值型周数"""
    if pd.isna(week_str):
        return None
    # 正则匹配周数和天数
    match = re.match(r'(\d+)w(?:\+(\d+))?', str(week_str))
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return round(weeks + days / 7.0, 3)  # 转换为总周数并保留3位小数
    return None

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    使用IQR方法去除异常值
    factor: 控制异常值检测的严格程度，通常为1.5
    """
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 保留在范围内的数据
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    print(f"异常值处理: 原始数据{len(df)}行，处理后{len(df_clean)}行")
    return df_clean

def denoise_data(df, columns, window_length=5, polyorder=2):
    """
    使用Savitzky-Golay滤波器去除数据噪声
    适用于平滑数据同时保留重要特征
    """
    df_denoised = df.copy()
    for col in columns:
        # 确保窗口长度小于数据长度
        wl = min(window_length, len(df_denoised) - 1)
        if wl % 2 == 0:  # 窗口长度必须是奇数
            wl -= 1
        if wl > polyorder:  # 多项式阶数必须小于窗口长度
            try:
                df_denoised[col] = savgol_filter(df_denoised[col], wl, polyorder)
            except:
                print(f"无法对列 {col} 应用滤波器，保持原值")
    
    print("数据去噪完成")
    return df_denoised

def standardize_data(df, feature_cols):
    """
    对特征数据进行标准化处理（无量纲化）
    将数据转换为均值为0，标准差为1的分布
    """
    df_standardized = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:  # 避免除以零
            df_standardized[col] = (df[col] - mean) / std
        else:
            df_standardized[col] = 0  # 如果标准差为0，全部置为0
    return df_standardized

def normalize_data(df, feature_cols):
    """
    对特征数据进行归一化处理（无量纲化）
    将数据转换到[0, 1]区间
    """
    df_normalized = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:  # 避免除以零
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0  # 如果所有值相同，全部置为0
    return df_normalized

def ensure_non_negative(df, columns):
    """
    确保数据非负，通过平移方法
    """
    df_non_neg = df.copy()
    for col in columns:
        min_val = df[col].min()
        if min_val < 0:
            df_non_neg[col] = df[col] - min_val  # 平移使最小值为0
    return df_non_neg

def load_and_preprocess_data(file_path, scaling_method='standardize', 
                           remove_noise=True, remove_outliers=True):
    """加载数据并进行预处理，包括去噪、异常值处理和无量纲化处理"""
    # 读取Excel数据
    df = pd.read_excel(file_path)
    
    # 确保孕妇代码列存在
    if '孕妇代码' not in df.columns:
        raise ValueError("数据集中必须包含'孕妇代码'列")
    
    # 转换孕周数据
    if '检测孕周' in df.columns:
        df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)
    else:
        raise ValueError("数据集中必须包含'检测孕周'列")
    
    # 选择需要的指标列
    target_col = 'Y染色体浓度'
    feature_cols = ['孕妇BMI', '年龄', '检测孕周数值', '在参考基因组上比对的比例']
    required_columns = ['孕妇代码', target_col] + feature_cols
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据集中缺少必要的列: {col}")
    
    # 保留需要的列并删除缺失值
    df_processed = df[required_columns].copy().dropna()
    print(f"缺失值处理: 原始数据{len(df)}行，处理后{len(df_processed)}行")
    
    # 异常值处理
    if remove_outliers:
        numeric_cols = [target_col] + feature_cols
        df_processed = remove_outliers_iqr(df_processed, numeric_cols)
    
    # 数据去噪
    if remove_noise:
        numeric_cols = [target_col] + feature_cols
        df_processed = denoise_data(df_processed, numeric_cols)
    
    # 执行无量纲化处理
    columns_to_scale = [target_col] + feature_cols
    if scaling_method == 'standardize':
        df_scaled = standardize_data(df_processed, columns_to_scale)
    elif scaling_method == 'normalize':
        df_scaled = normalize_data(df_processed, columns_to_scale)
    else:
        df_scaled = df_processed.copy()
        print("未进行无量纲化处理")
    
    # 确保数据非负（灰色模型要求）
    df_scaled = ensure_non_negative(df_scaled, columns_to_scale)
    
    print(f"数据预处理完成: 最终数据{len(df_scaled)}行")
    return df_scaled, target_col, feature_cols

# 2. 蒙特卡洛抽样函数
def monte_carlo_sample(df, sample_size=None):
    """
    对每个孕妇随机抽取一条数据
    sample_size: 目标样本量，None则使用所有可能的孕妇
    """
    # 按孕妇代码分组
    grouped = df.groupby('孕妇代码')
    
    # 对每个孕妇随机选择一条记录
    selected = []
    for _, group in grouped:
        # 随机选择一条记录
        selected.append(group.sample(n=1, random_state=np.random.randint(0, 10000)))
    
    # 合并选择的记录
    sampled_df = pd.concat(selected).reset_index(drop=True)
    
    # 如果指定了样本量且实际样本量超过，则进一步抽样
    if sample_size and len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=np.random.randint(0, 10000))
    
    return sampled_df

# 3. 灰色预测模型(GM(1,N))实现及模型评估指标
def calculate_grey_evaluation_metrics(X0, X0_fit):
    """
    计算灰色预测模型的评估指标
    X0: 原始序列
    X0_fit: 拟合序列
    返回:
        C: 后验差比
        P: 小误差概率
        e: 残差序列
    """
    n = len(X0)
    # 计算残差
    e = X0 - X0_fit
    
    # 计算原始序列的均值和标准差(使用无偏估计)
    x_mean = np.mean(X0)
    s0 = np.sqrt(np.sum((X0 - x_mean) ** 2) / (n - 1)) if n > 1 else 0
    
    # 计算残差的均值和标准差(使用无偏估计)
    e_mean = np.mean(e)
    s1 = np.sqrt(np.sum((e - e_mean) ** 2) / (n - 1)) if n > 1 else 0
    
    # 计算后验差比 C
    C = s1 / s0 if s0 != 0 else 0
    
    # 计算小误差概率 P
    threshold = 0.6745 * s0
    P = np.mean(np.abs(e - e_mean) < threshold)
    
    return C, P, e

def gm1n_model(X0, X, verbose=False):
    """
    实现GM(1,N)灰色预测模型
    X0: 母序列（Y染色体浓度）
    X: 子序列（影响因素）
    返回模型参数(a, b)、拟合值和评估指标(C, P)
    """
    n = len(X0)
    m = X.shape[1]  # 子序列数量
    
    # 检查数据是否满足建模要求
    if n < 5:  # 至少需要5个数据点
        if verbose:
            print("数据点不足，无法构建模型")
        return None, None, None, None, None, None
        
    # 检查数据是否非负
    if np.any(X0 < 0) or np.any(X < 0):
        if verbose:
            print("数据包含负值，不符合灰色模型要求")
        return None, None, None, None, None, None
    
    # 1-AGO 一次累加生成
    X0_ago = np.cumsum(X0)
    X_agos = np.array([np.cumsum(X[:, i]) for i in range(m)]).T
    
    # 构造B矩阵和Y矩阵
    B = np.zeros((n-1, m+1))
    Y = np.zeros((n-1, 1))
    
    for k in range(1, n):
        # 母序列背景值
        Z0 = 0.5 * X0_ago[k] + 0.5 * X0_ago[k-1]
        Y[k-1, 0] = X0[k]
        B[k-1, 0] = -Z0
        
        # 子序列背景值
        for i in range(m):
            Zi = 0.5 * X_agos[k, i] + 0.5 * X_agos[k-1, i]
            B[k-1, i+1] = Zi
    
    # 最小二乘法求解参数
    try:
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        a = params[0, 0]  # 发展系数
        b = params[1:, 0]  # 驱动系数
        
        # 模型拟合
        X0_fit = np.zeros_like(X0)
        X0_fit[0] = X0[0]
        
        for k in range(1, n):
            Z0 = 0.5 * X0_ago[k] + 0.5 * X0_ago[k-1]
            Zi_list = [0.5 * X_agos[k, i] + 0.5 * X_agos[k-1, i] for i in range(m)]
            X0_fit[k] = -a * Z0 + np.sum(b * np.array(Zi_list))
        
        # 计算模型评估指标：后验差比C和小误差概率P
        C, P, _ = calculate_grey_evaluation_metrics(X0, X0_fit)
        
        # 计算拟合误差
        residuals = X0 - X0_fit
        rmse = np.sqrt(np.mean(residuals**2))
        
        if verbose:
            print(f"GM(1,N)模型构建完成 - 发展系数a: {a:.6f}, 驱动系数b: {b}")
            print(f"模型评估 - 后验差比C: {C:.6f}, 小误差概率P: {P:.6f}")
        
        return a, b, X0_fit, rmse, C, P
    except np.linalg.LinAlgError:
        # 处理矩阵奇异的情况
        if verbose:
            print("矩阵奇异，无法求解模型参数")
        return None, None, None, None, None, None

# 4. 蒙特卡洛模拟主函数
def run_monte_carlo_simulation(df, target_col, feature_cols, n_simulations=500, 
                              sample_size=100, save_interval=100):
    """
    运行蒙特卡洛模拟
    n_simulations: 模拟次数
    sample_size: 每次抽样的样本量
    save_interval: 每隔多少次模拟保存一次中间结果
    """
    # 存储所有模拟的模型参数和评估指标
    results = {
        'a': [],  # 发展系数
        'b': [],  # 驱动系数
        'rmse': [],  # 模型RMSE
        'C': [],  # 后验差比
        'P': [],  # 小误差概率
        'sample_size': []  # 实际样本量
    }
    
    # 创建结果保存目录
    if not os.path.exists('simulation_results'):
        os.makedirs('simulation_results')
    
    # 运行模拟
    for i in tqdm(range(n_simulations), desc="运行蒙特卡洛模拟"):
        # 1. 蒙特卡洛抽样
        sampled_df = monte_carlo_sample(df, sample_size)
        
        # 2. 准备建模数据
        X0 = sampled_df[target_col].values
        X = sampled_df[feature_cols].values
        
        # 确保有足够的样本进行建模
        if len(sampled_df) < 10:  # 至少需要10个样本
            continue
        
        # 3. 构建GM(1,N)模型
        a, b, _, rmse, C, P = gm1n_model(X0, X)
        
        # 4. 保存有效结果
        if a is not None and b is not None and rmse is not None and C is not None and P is not None:
            results['a'].append(a)
            results['b'].append(b)
            results['rmse'].append(rmse)
            results['C'].append(C)
            results['P'].append(P)
            results['sample_size'].append(len(sampled_df))
        
        # 定期保存中间结果
        if (i + 1) % save_interval == 0:
            with open(f'simulation_results/results_{i+1}.pkl', 'wb') as f:
                pickle.dump(results, f)
    
    # 保存最终结果
    with open('simulation_results/final_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"蒙特卡洛模拟完成 - 有效模型数量: {len(results['a'])}/{n_simulations}")
    return results

# 5. 结果分析与可视化
def analyze_results(results, feature_cols, confidence_interval=0.95):
    """分析模拟结果并可视化，包括打印后验差比(C值)和小误差概率(P值)"""
    # 1. 计算参数的统计值
    a_values = np.array(results['a'])
    b_values = np.array(results['b'])
    rmse_values = np.array(results['rmse'])
    C_values = np.array(results['C'])  # 后验差比
    P_values = np.array(results['P'])  # 小误差概率
    
    # 计算均值和标准差
    a_mean = np.mean(a_values)
    a_std = np.std(a_values)
    
    # 计算每个驱动系数的均值和标准差
    b_means = np.mean(b_values, axis=0)
    b_stds = np.std(b_values, axis=0)
    
    # 2. 计算模型评估指标的统计值
    C_mean = np.mean(C_values)
    C_std = np.std(C_values)
    P_mean = np.mean(P_values)
    P_std = np.std(P_values)
    
    # 3. 打印模型评估指标（C值和P值）
    print("\n=== 模型评估指标统计结果 ===")
    print(f"后验差比(C值): 均值 = {C_mean:.6f}, 标准差 = {C_std:.6f}")
    print(f"小误差概率(P值): 均值 = {P_mean:.6f}, 标准差 = {P_std:.6f}")
    
    # 解释模型精度等级（根据灰色系统理论）
    print("\n=== 模型精度等级评估 ===")
    if C_mean < 0.35 and P_mean > 0.95:
        print("模型精度等级：一级（优）")
    elif C_mean < 0.5 and P_mean > 0.8:
        print("模型精度等级：二级（良）")
    elif C_mean < 0.65 and P_mean > 0.7:
        print("模型精度等级：三级（中）")
    else:
        print("模型精度等级：四级（差）")
    
    # 4. 打印模型参数统计结果
    print("\n=== 模型参数统计结果 ===")
    print(f"发展系数a: 均值 = {a_mean:.6f}, 标准差 = {a_std:.6f}")
    for i, col in enumerate(feature_cols):
        print(f"驱动系数b[{i}] ({col}): 均值 = {b_means[i]:.6f}, 标准差 = {b_stds[i]:.6f}")
    
    print(f"\n模型平均RMSE: {np.mean(rmse_values):.6f}")
    
    # 5. 可视化参数分布
    plt.figure(figsize=(18, 12))
    
    # 发展系数a的分布
    plt.subplot(3, 3, 1)
    sns.histplot(a_values, kde=True)
    plt.axvline(a_mean, color='r', linestyle='--', label=f'均值: {a_mean:.6f}')
    plt.title('发展系数a的分布')
    plt.xlabel('a值')
    plt.ylabel('频数')
    plt.legend()
    
    # 驱动系数b的分布
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 3, i + 2)
        sns.histplot(b_values[:, i], kde=True)
        plt.axvline(b_means[i], color='r', linestyle='--', label=f'均值: {b_means[i]:.6f}')
        plt.title(f'驱动系数b[{i}] ({col})的分布')
        plt.xlabel('b值')
        plt.ylabel('频数')
        plt.legend()
    
    # 后验差比C的分布
    plt.subplot(3, 3, 6)
    sns.histplot(C_values, kde=True)
    plt.axvline(C_mean, color='r', linestyle='--', label=f'均值: {C_mean:.6f}')
    plt.title('后验差比(C值)的分布')
    plt.xlabel('C值')
    plt.ylabel('频数')
    plt.legend()
    
    # 小误差概率P的分布
    plt.subplot(3, 3, 7)
    sns.histplot(P_values, kde=True)
    plt.axvline(P_mean, color='r', linestyle='--', label=f'均值: {P_mean:.6f}')
    plt.title('小误差概率(P值)的分布')
    plt.xlabel('P值')
    plt.ylabel('频数')
    plt.legend()
    
    # RMSE的分布
    plt.subplot(3, 3, 8)
    sns.histplot(rmse_values, kde=True)
    plt.axvline(np.mean(rmse_values), color='r', linestyle='--', label=f'均值: {np.mean(rmse_values):.6f}')
    plt.title('RMSE的分布')
    plt.xlabel('RMSE值')
    plt.ylabel('频数')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_results/parameter_distributions_cp2.png')
    plt.show()
    
    # 6. 可视化参数收敛情况
    plt.figure(figsize=(15, 12))
    
    # 发展系数a的累积均值
    plt.subplot(3, 2, 1)
    cumulative_a_mean = np.cumsum(a_values) / np.arange(1, len(a_values) + 1)
    plt.plot(cumulative_a_mean)
    plt.axhline(a_mean, color='r', linestyle='--', label=f'最终均值: {a_mean:.6f}')
    plt.title('发展系数a的累积均值变化')
    plt.xlabel('模拟次数')
    plt.ylabel('累积均值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 驱动系数b的累积均值
    plt.subplot(3, 2, 2)
    for i, col in enumerate(feature_cols):
        cumulative_b_mean = np.cumsum(b_values[:, i]) / np.arange(1, len(b_values) + 1)
        plt.plot(cumulative_b_mean, label=f'{col}: {b_means[i]:.6f}')
    plt.title('驱动系数b的累积均值变化')
    plt.xlabel('模拟次数')
    plt.ylabel('累积均值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 后验差比C的累积均值
    plt.subplot(3, 2, 3)
    cumulative_C_mean = np.cumsum(C_values) / np.arange(1, len(C_values) + 1)
    plt.plot(cumulative_C_mean)
    plt.axhline(C_mean, color='r', linestyle='--', label=f'最终均值: {C_mean:.6f}')
    plt.title('后验差比(C值)的累积均值变化')
    plt.xlabel('模拟次数')
    plt.ylabel('累积均值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 小误差概率P的累积均值
    plt.subplot(3, 2, 4)
    cumulative_P_mean = np.cumsum(P_values) / np.arange(1, len(P_values) + 1)
    plt.plot(cumulative_P_mean)
    plt.axhline(P_mean, color='r', linestyle='--', label=f'最终均值: {P_mean:.6f}')
    plt.title('小误差概率(P值)的累积均值变化')
    plt.xlabel('模拟次数')
    plt.ylabel('累积均值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RMSE的累积均值
    plt.subplot(3, 2, 5)
    cumulative_rmse_mean = np.cumsum(rmse_values) / np.arange(1, len(rmse_values) + 1)
    plt.plot(cumulative_rmse_mean)
    plt.axhline(np.mean(rmse_values), color='r', linestyle='--', label=f'最终均值: {np.mean(rmse_values):.6f}')
    plt.title('RMSE的累积均值变化')
    plt.xlabel('模拟次数')
    plt.ylabel('累积均值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results/parameter_convergence_cp2.png')
    plt.show()
    
    # 7. 输出最终收敛的模型表达式
    print("\n=== 收敛的GM(1,N)模型表达式 ===")
    equation = f"X⁽⁰⁾(k) = {-a_mean:.6f}·Z⁽¹⁾₀(k) "
    for i, col in enumerate(feature_cols):
        sign = "+" if b_means[i] >= 0 else "-"
        equation += f"{sign} {abs(b_means[i]):.6f}·Z⁽¹⁾_{i+1}(k) "
    print(equation)
    
    return {
        'a_mean': a_mean,
        'a_std': a_std,
        'b_means': b_means,
        'b_stds': b_stds,
        'C_mean': C_mean,
        'C_std': C_std,
        'P_mean': P_mean,
        'P_std': P_std,
        'rmse_mean': np.mean(rmse_values),
        'equation': equation
    }

# 6. 主函数
def main():
    # 配置参数
    n_simulations = 1000  # 模拟次数
    sample_size = 200     # 每次抽样的目标样本量
    data_path = './data/data-total.xlsx'  # 数据文件路径
    scaling_method = 'standardize'  # 无量纲化方法: 'standardize' 或 'normalize'
    
    # 执行流程
    try:
        # 1. 加载和预处理数据（包含去噪、异常值处理和无量纲化）
        df, target_col, feature_cols = load_and_preprocess_data(
            data_path, 
            scaling_method=scaling_method,
            remove_noise=True,
            remove_outliers=True
        )
        
        # 2. 运行蒙特卡洛模拟
        results = run_monte_carlo_simulation(
            df, 
            target_col,
            feature_cols,
            n_simulations=n_simulations,
            sample_size=sample_size
        )
        
        # 3. 分析结果并输出最终模型，包括后验差比(C值)和小误差概率(P值)
        final_model = analyze_results(results, feature_cols)
        
        # 4. 保存最终模型
        with open('simulation_results/final_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)
            
        print("\n分析完成，结果已保存到'simulation_results'目录")
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()