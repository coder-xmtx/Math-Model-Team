import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from keras import Sequential
import keras
import matplotlib.pyplot as plt
import joblib  # 用于保存标准化器
import os

# 设置中文显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建目录保存模型和标准化器
os.makedirs('models', exist_ok=True)

# 读取数据
data = pd.read_excel('data/data-girl.xlsx')

# 数据预处理
def convert_gestational_week(week_str):
    try:
        if '+' in week_str:
            parts = week_str.split('w+')
            week = float(parts[0])
            day = float(parts[1])
            return week + day / 7.0
        else:
            return float(week_str.replace('w', ''))
    except:
        return np.nan

data['检测孕周'] = data['检测孕周'].apply(convert_gestational_week)
data['异常标签'] = data['染色体的非整倍体'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)

features = ['年龄', '身高', '体重', '检测孕周', '孕妇BMI', '在参考基因组上比对的比例', 
            '重复读段的比例', 'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
X = data[features]
y = data['异常标签']

# 处理缺失值
X.fillna(X.mean(), inplace=True)
y.fillna(0, inplace=True)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 保存标准化器
joblib.dump(scaler, 'models/scaler.pkl')
print("标准化器已保存为 models/scaler.pkl")

# 初始化结果存储
accuracies = []
recalls = []
f1_scores = []
aucs = []
feature_importances = []
n_iterations = 100

# 蒙特卡洛模拟
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i, stratify=y)
    
    model = Sequential([
        keras.layers.Dense(24, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    class_weight = {0: 1., 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, class_weight=class_weight)
    
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    aucs.append(auc(fpr, tpr))
    
    weights = model.layers[0].get_weights()[0]
    feature_importances.append(np.mean(np.abs(weights), axis=1))

# 计算平均性能指标和特征重要性
mean_accuracy = np.mean(accuracies)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_auc = np.mean(aucs)
mean_feature_importance = np.mean(feature_importances, axis=0)

# 训练最终模型（使用全部数据）
final_model = Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(X_scaled.shape[1],)),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 计算最终模型的类别权重
final_class_weight = {0: 1., 1: len(y[y==0]) / len(y[y==1])}

# 训练最终模型
final_model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=0, class_weight=final_class_weight)

# 保存最终模型
final_model.save('models/final_model.h5')
print("最终模型已保存为 models/final_model.h5")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(features, mean_feature_importance)
plt.xlabel('平均特征重要性')
plt.title('特征重要性排序')
plt.tight_layout()
plt.savefig('models/feature_importance.png')  # 保存特征重要性图
plt.show()

# 绘制平均ROC曲线
plt.figure(figsize=(10, 6))
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i, stratify=y)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, class_weight=class_weight)
    y_pred_prob = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, alpha=0.1, color='grey')

mean_tpr = np.linspace(0, 1, 100)
mean_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'平均ROC曲线 (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend()
plt.savefig('models/roc_curve.png')  # 保存ROC曲线图
plt.show()

# 绘制召回率-精确率曲线
plt.figure(figsize=(10, 6))
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i, stratify=y)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, class_weight=class_weight)
    y_pred_prob = model.predict(X_test).flatten()
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall, precision, alpha=0.1, color='grey')

mean_precision = np.mean(precision)
mean_recall = np.mean(recall)
plt.plot(recall, precision, color='blue', label='平均召回率-精确率曲线')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('召回率-精确率曲线')
plt.legend()
plt.savefig('models/precision_recall_curve.png')  # 保存精确率-召回率曲线图
plt.show()

# 输出结果
print(f"平均准确率: {mean_accuracy:.4f}")
print(f"平均召回率: {mean_recall:.4f}")
print(f"平均F1分数: {mean_f1:.4f}")
print(f"平均AUC: {mean_auc:.4f}")
print("\n特征重要性排序:")
for feature, importance in zip(features, mean_feature_importance):
    print(f"{feature}: {importance:.4f}")

# 保存性能指标到文件
with open('models/performance_metrics.txt', 'w') as f:
    f.write(f"平均准确率: {mean_accuracy:.4f}\n")
    f.write(f"平均召回率: {mean_recall:.4f}\n")
    f.write(f"平均F1分数: {mean_f1:.4f}\n")
    f.write(f"平均AUC: {mean_auc:.4f}\n")
    f.write("\n特征重要性排序:\n")
    for feature, importance in zip(features, mean_feature_importance):
        f.write(f"{feature}: {importance:.4f}\n")

print("所有模型和结果已保存到 models/ 目录")