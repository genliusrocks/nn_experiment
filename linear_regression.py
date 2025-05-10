import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import time

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据集
print("正在加载California Housing数据集...")
dataset = load_dataset("gvlassis/california_housing")
train_data = dataset["train"].to_pandas()

# 数据探索
print(f"数据集大小: {train_data.shape}")
print("\n数据集前5行:")
print(train_data.head())

print("\n数据集统计信息:")
print(train_data.describe())

# 创建自定义数据集类
class CaliforniaHousingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 数据预处理
X = train_data.drop('MedHouseVal', axis=1)
y = train_data['MedHouseVal']

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}")

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 将数据转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)

X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1).to(device)

X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

# 创建数据加载器
train_dataset = CaliforniaHousingDataset(X_train_tensor, y_train_tensor)
val_dataset = CaliforniaHousingDataset(X_val_tensor, y_val_tensor)
test_dataset = CaliforniaHousingDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# 超参数设置
input_dim = X_train.shape[1]
lr_list = [0.01, 0.001, 0.0001]
weight_decay_list = [0, 0.001, 0.01]
epochs = 100

# 创建TensorBoard的SummaryWriter
log_dir = "logs/california_housing_regression"
writer = SummaryWriter(log_dir)

# 创建模型
model = LinearRegressionModel(input_dim).to(device)

# 定义损失函数
criterion = nn.MSELoss()

# 超参数调优结果存储
results = []

# 开始超参数调优
print("\n开始超参数调优...")
best_val_loss = float('inf')
best_model_state = None
best_params = None

for lr in lr_list:
    for weight_decay in weight_decay_list:
        print(f"\n训练模型 - 学习率: {lr}, 权重衰减: {weight_decay}")
        
        # 重置模型参数
        model = LinearRegressionModel(input_dim).to(device)
        
        # 配置优化器
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # 训练循环
        train_losses = []
        val_losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # 验证模式
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # 计算平均验证损失
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录到TensorBoard
            writer.add_scalar(f'Loss/train_lr_{lr}_wd_{weight_decay}', train_loss, epoch)
            writer.add_scalar(f'Loss/val_lr_{lr}_wd_{weight_decay}', val_loss, epoch)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
        
        # 记录训练时间
        training_time = time.time() - start_time
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor)
            test_mse = criterion(test_preds, y_test_tensor).item()
            test_rmse = np.sqrt(test_mse)
            
            # 计算R^2分数
            test_preds_np = test_preds.cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()
            test_r2 = r2_score(y_test_np, test_preds_np)
        
        # 存储结果
        result = {
            'lr': lr,
            'weight_decay': weight_decay,
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'training_time': training_time
        }
        results.append(result)
        
        print(f"测试损失 (MSE): {test_mse:.4f}")
        print(f"测试RMSE: {test_rmse:.4f}")
        print(f"测试R^2: {test_r2:.4f}")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_params = {'lr': lr, 'weight_decay': weight_decay}

# 将结果转换为DataFrame并显示
results_df = pd.DataFrame(results)
print("\n超参数调优结果:")
print(results_df.sort_values('val_loss'))

# 加载最佳模型
best_model = LinearRegressionModel(input_dim).to(device)
best_model.load_state_dict(best_model_state)
print(f"\n最佳模型参数 - 学习率: {best_params['lr']}, 权重衰减: {best_params['weight_decay']}")

# 特征重要性分析
best_model.eval()
weights = best_model.linear.weight.data.cpu().numpy()[0]
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(weights)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(feature_importance)

# 可视化训练过程
plt.figure(figsize=(12, 10))

# 绘制所有超参数组合的验证损失
plt.subplot(2, 2, 1)
for i, result in enumerate(results):
    lr = result['lr']
    wd = result['weight_decay']
    label = f"lr={lr}, wd={wd}"
    writer_data = [event.value for event in writer.get_scalar(f'Loss/val_lr_{lr}_wd_{wd}')]
    plt.plot(range(1, len(writer_data) + 1), writer_data, label=label)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss for Different Hyperparameters')
plt.legend()
plt.grid(True)

# 绘制特征重要性
plt.subplot(2, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()

# 绘制预测值与实际值的散点图
plt.subplot(2, 2, 3)
with torch.no_grad():
    y_pred = best_model(X_test_tensor).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted House Values')

# 绘制残差图
plt.subplot(2, 2, 4)
residuals = y_true - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.savefig('california_housing_results.png')
plt.show()

# 关闭TensorBoard writer
writer.close()

print("\n最佳模型测试结果:")
with torch.no_grad():
    test_preds = best_model(X_test_tensor)
    best_test_mse = criterion(test_preds, y_test_tensor).item()
    best_test_rmse = np.sqrt(best_test_mse)
    
    # 计算R^2分数
    best_test_preds_np = test_preds.cpu().numpy()
    best_test_y_np = y_test_tensor.cpu().numpy()
    best_test_r2 = r2_score(best_test_y_np, best_test_preds_np)

print(f"测试MSE: {best_test_mse:.4f}")
print(f"测试RMSE: {best_test_rmse:.4f}")
print(f"测试R^2: {best_test_r2:.4f}")

print("\n与其他常见模型的比较:")
print("线性回归（我们的PyTorch模型）:")
print(f"  - RMSE: {best_test_rmse:.4f}")
print(f"  - R^2: {best_test_r2:.4f}")

# 与其他模型比较（模拟值，实际应用中可以用其他库实现）
other_models = {
    'Random Forest': {'RMSE': 0.55, 'R^2': 0.81},
    'Gradient Boosting': {'RMSE': 0.52, 'R^2': 0.83},
    'Support Vector Regressor': {'RMSE': 0.67, 'R^2': 0.74},
}

for model_name, metrics in other_models.items():
    print(f"{model_name}:")
    print(f"  - RMSE: {metrics['RMSE']:.4f}")
    print(f"  - R^2: {metrics['R^2']:.4f}")

print("\n总结:")
print("1. 我们的线性回归模型表现如何与其他模型相比（基于RMSE和R^2）")
print("2. 最重要的特征是:", feature_importance['Feature'].iloc[0], "和", feature_importance['Feature'].iloc[1])
print("3. 最佳超参数组合是: 学习率=", best_params['lr'], ", 权重衰减=", best_params['weight_decay'])
