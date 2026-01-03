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
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import shap

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

def create_interaction_features(df, feature_cols, degree=2):
    """创建交互特征和多项式特征"""
    # 选择数值型特征
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[numeric_features])
    poly_feature_names = poly.get_feature_names_out(numeric_features)
    
    # 创建交互特征数据框
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    
    # 添加原始特征和目标变量
    for col in df.columns:
        if col not in numeric_features and col != 'Y染色体浓度':
            poly_df[col] = df[col].values
    
    poly_df['Y染色体浓度'] = df['Y染色体浓度'].values
    
    return poly_df, poly_feature_names.tolist()

def select_features(df, feature_cols, target_col, k=10, method='mutual_info'):
    """选择最佳特征"""
    X = df[feature_cols]
    y = df[target_col]
    
    # 选择特征选择方法
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(feature_cols)))
    else:  # f_regression
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
    
    selector.fit(X, y)
    
    # 获取选择的特征
    selected_mask = selector.get_support()
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
    
    # 获取特征得分
    feature_scores = pd.DataFrame({
        'feature': feature_cols,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print(f"选择了 {len(selected_features)} 个最佳特征:")
    for _, row in feature_scores.head(k).iterrows():
        print(f"  - {row['feature']}: {row['score']:.6f}")
    
    return selected_features, feature_scores

def load_and_preprocess_data(file_path, remove_noise=True, remove_outliers=True, create_poly_features=True):
    """加载数据并进行预处理"""
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
    base_feature_cols = ['孕妇BMI', '年龄', '检测孕周数值', '在参考基因组上比对的比例']
    required_columns = ['孕妇代码', target_col] + base_feature_cols
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据集中缺少必要的列: {col}")
    
    # 保留需要的列并删除缺失值
    df_processed = df[required_columns].copy().dropna()
    print(f"缺失值处理: 原始数据{len(df)}行，处理后{len(df_processed)}行")
    
    # 异常值处理
    if remove_outliers:
        numeric_cols = [target_col] + base_feature_cols
        df_processed = remove_outliers_iqr(df_processed, numeric_cols)
    
    # 数据去噪
    if remove_noise:
        numeric_cols = [target_col] + base_feature_cols
        df_processed = denoise_data(df_processed, numeric_cols)
    
    # 创建多项式特征
    if create_poly_features:
        df_processed, poly_feature_names = create_interaction_features(df_processed, base_feature_cols, degree=2)
        feature_cols = poly_feature_names
    else:
        feature_cols = base_feature_cols
    
    print(f"数据预处理完成: 最终数据{len(df_processed)}行, 特征数量: {len(feature_cols)}")
    return df_processed, target_col, feature_cols

# 2. 模型训练与评估函数
def train_xgboost_model(X_train, y_train, X_test, y_test, param_grid=None):
    """训练XGBoost模型并返回最佳模型和评估结果"""
    # 默认参数
    if param_grid is None:
        param_grid = {
            'n_estimators': [50,100],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree':[0.8,0.9],
            'tree_method':['hist']
        }
    
    # 创建基础模型
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    # 网格搜索寻找最佳参数
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 交叉验证
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"测试集 RMSE: {rmse:.6f}")
    print(f"测试集 MAE: {mae:.6f}")
    print(f"测试集 R²: {r2:.6f}")
    print(f"交叉验证 RMSE: {cv_rmse.mean():.6f} (±{cv_rmse.std():.6f})")
    
    return best_model, rmse, mae, r2, cv_rmse

def compare_models(X_train, y_train, X_test, y_test):
    """比较不同模型的性能"""
    models = {
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42)
    }
    
    results = {}
    print("=== 模型性能比较 ===")
    
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
        
        print(f"{name}:")
        print(f"  RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        print(f"  交叉验证 RMSE: {cv_rmse.mean():.6f} (±{cv_rmse.std():.6f})")
    
    return results

# 3. 结果分析与可视化
def plot_feature_importance(model, feature_names, top_n=10):
    """绘制特征重要性图"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title("特征重要性")
    plt.barh(range(min(top_n, len(indices))), importance[indices][:top_n][::-1])
    plt.yticks(range(min(top_n, len(indices))), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel("重要性")
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.show()
    
    # 返回特征重要性数据框
    feature_importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importance[indices]
    })
    
    return feature_importance_df

def plot_shap_summary(model, X, feature_names):
    """绘制SHAP摘要图"""
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 摘要图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary.png')
    plt.show()
    
    # 计算平均绝对SHAP值
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    return shap_df

def plot_residuals(y_true, y_pred):
    """绘制残差图"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 残差分布
    axes[0].hist(residuals, bins=30, alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--')
    axes[0].set_xlabel('残差')
    axes[0].set_ylabel('频数')
    axes[0].set_title('残差分布')
    
    # 残差 vs 预测值
    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差 vs 预测值')
    
    plt.tight_layout()
    plt.savefig('results/residual_analysis.png')
    plt.show()

# 4. 主函数
def main():
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 配置参数
    data_path = './data/data-total.xlsx'  # 数据文件路径
    
    try:
        # 1. 加载和预处理数据
        print("加载和预处理数据...")
        df, target_col, feature_cols = load_and_preprocess_data(
            data_path, 
            remove_noise=True,
            remove_outliers=True,
            create_poly_features=True
        )
        
        # 2. 特征选择
        print("\n进行特征选择...")
        selected_features, feature_scores = select_features(
            df, feature_cols, target_col, k=15, method='mutual_info'
        )
        
        # 3. 准备数据
        X = df[selected_features].values
        y = df[target_col].values
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 4. 比较不同模型
        print("\n比较不同模型性能...")
        model_results = compare_models(X_train, y_train, X_test, y_test)
        
        # 5. 训练XGBoost模型（使用最佳参数）
        print("\n训练XGBoost模型...")
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        best_model, rmse, mae, r2, cv_rmse = train_xgboost_model(
            X_train, y_train, X_test, y_test, xgb_params
        )
        
        # 6. 特征重要性分析
        print("\n分析特征重要性...")
        feature_importance_df = plot_feature_importance(best_model, selected_features, top_n=15)
        
        # 7. SHAP分析
        print("\n进行SHAP分析...")
        shap_df = plot_shap_summary(best_model, X_test, selected_features)
        
        # 8. 残差分析
        print("\n进行残差分析...")
        y_pred = best_model.predict(X_test)
        plot_residuals(y_test, y_pred)
        
        # 9. 保存模型和结果
        print("\n保存模型和结果...")
        # 保存模型
        best_model.save_model('results/xgboost_model.json')
        
        # 保存特征重要性
        feature_importance_df.to_csv('results/feature_importance.csv', index=False)
        feature_scores.to_csv('results/feature_scores.csv', index=False)
        shap_df.to_csv('results/shap_values.csv', index=False)
        
        # 保存结果摘要
        results_summary = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'selected_features': selected_features,
            'feature_importance': feature_importance_df.to_dict(),
            'model_comparison': model_results
        }
        
        with open('results/results_summary.pkl', 'wb') as f:
            pickle.dump(results_summary, f)
        
        # 10. 输出最终结果
        print("\n=== 最终结果 ===")
        print(f"测试集 RMSE: {rmse:.6f}")
        print(f"测试集 MAE: {mae:.6f}")
        print(f"测试集 R²: {r2:.6f}")
        print(f"交叉验证 RMSE: {cv_rmse.mean():.6f} (±{cv_rmse.std():.6f})")
        
        print("\n=== 最重要的特征 ===")
        for i, row in feature_importance_df.head(5).iterrows():
            print(f"{row['feature']}: {row['importance']:.6f}")
        
        print("\n分析完成，结果已保存到'results'目录")
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()