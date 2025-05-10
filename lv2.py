# California Housing Regression with PyTorch and HuggingFace Datasets
# (Refactored code meeting specified requirements)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# 定义一个 Dataset 包装 HuggingFace Dataset，以返回 (features, target) 对
class CaliforniaHousingDataset(Dataset):
    def __init__(self, hf_dataset, feature_names, target_name):
        self.dataset = hf_dataset
        self.feature_names = feature_names
        self.target_name = target_name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 获取 HuggingFace 数据集中索引为 idx 的数据
        data = self.dataset[idx]  # 返回一个字典，包含该样本的所有字段
        # 提取特征和目标值
        features = [data[name] for name in self.feature_names]
        X = torch.tensor(features, dtype=torch.float32)             # 将特征转换为 float32 张量
        y = torch.tensor(data[self.target_name], dtype=torch.float32)  # 将目标值转换为 float32 张量
        return X, y

# 加载 California Housing 数据集的 train/validation/test 划分
dataset = load_dataset("gvlassis/california_housing")
train_hf = dataset["train"]
val_hf   = dataset["validation"]
test_hf  = dataset["test"]

# 特征列名称（排除目标列）
feature_names = [col for col in train_hf.column_names if col != "MedHouseVal"]
target_name   = "MedHouseVal"

# 使用自定义 Dataset 包装 HuggingFace 数据集
train_dataset = CaliforniaHousingDataset(train_hf, feature_names, target_name)
val_dataset   = CaliforniaHousingDataset(val_hf, feature_names, target_name)
test_dataset  = CaliforniaHousingDataset(test_hf, feature_names, target_name)

# 定义一个简单的多层感知机模型用于回归任务
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 超参数设置
learning_rate = 1e-3
num_epochs    = 50
batch_sizes   = [32, 64, 128]  # 不同的 batch_size 组合，用于对比训练效果

# 定义损失函数 (均方误差损失)
criterion = nn.MSELoss()

# 遍历不同的 batch_size 进行训练和评估
for batch_size in batch_sizes:
    # 固定随机种子，确保不同 batch_size 模型初始权重相同，便于公平对比
    torch.manual_seed(0)
    print(f"\n===== 开始训练：batch_size = {batch_size} =====")
    
    # 数据加载器 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型和优化器（每次更换 batch_size 重新初始化模型参数）
    model = RegressionModel(input_dim=len(feature_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # 前向传播
            preds = model(X_batch).squeeze()  # 模型预测输出，shape: (batch,)
            loss = criterion(preds, y_batch)   # 计算本批次的 MSE 损失
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)  # 累积损失之和（用于计算平均损失）
        # 计算整个训练集上的平均损失
        train_loss = total_loss / len(train_dataset)
        
        # 验证阶段（不启用梯度计算）
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze()
                loss = criterion(preds, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        val_loss = total_val_loss / len(val_dataset)
        
        # 每隔若干 epoch 输出一次训练和验证损失（这里每10个epoch输出一次，最后一个epoch也输出）
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:2d}/{num_epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
    
    # 当前 batch_size 下模型训练结束，计算在测试集上的最终 MSE
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            total_test_loss += loss.item() * X_batch.size(0)
    test_loss = total_test_loss / len(test_dataset)
    print(f"batch_size = {batch_size} 的训练完成！最终测试集 MSE: {test_loss:.4f}")

