import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np

# 1. 加载 HuggingFace 上的 train/validation/test 划分
dataset   = load_dataset("gvlassis/california_housing")
train_hf  = dataset["train"]
val_hf    = dataset["validation"]
test_hf   = dataset["test"]

# 2. 特征列与目标列名称
feature_names = [c for c in train_hf.column_names if c != "MedHouseVal"]
target_name   = "MedHouseVal"

# 3. 在训练集上计算每个 feature 的均值和标准差
# train_array的shape是(m, n)，其中m是样本数量，n是特征数量
train_array    = np.stack([train_hf[col] for col in feature_names], axis=1).astype(np.float32)
feature_means  = train_array.mean(axis=0)
feature_stds   = train_array.std(axis=0)

# 4. 自定义 Dataset：在 __getitem__ 中做标准化
class CaliforniaHousingDataset(Dataset):
    def __init__(self, hf_dataset, feature_names, target_name, means, stds):
        self.dataset       = hf_dataset
        self.feature_names = feature_names
        self.target_name   = target_name
        self.means         = means
        self.stds          = stds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        # 提取原始特征并做标准化
        raw = np.array([row[f] for f in self.feature_names], dtype=np.float32)
        norm = (raw - self.means) / self.stds
        X    = torch.from_numpy(norm)                                 # 标准化后特征张量
        y    = torch.tensor(row[self.target_name], dtype=torch.float32)  # 目标值张量
        return X, y

# 5. 构造 PyTorch Dataset 与 DataLoader
train_ds  = CaliforniaHousingDataset(train_hf, feature_names, target_name, feature_means, feature_stds)
val_ds    = CaliforniaHousingDataset(val_hf,   feature_names, target_name, feature_means, feature_stds)
test_ds   = CaliforniaHousingDataset(test_hf,  feature_names, target_name, feature_means, feature_stds)

# 6. 定义简单回归模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        #TBD
        return self.net(x)

# 7. 超参数
learning_rate = 1e-3
num_epochs    = 50
batch_sizes   = [32, 64, 128]

criterion = nn.MSELoss()

# 8. 对不同 batch_size 进行训练对比
for batch_size in batch_sizes:
    torch.manual_seed(0)
    print(f"\n>>> batch_size = {batch_size} 开始训练")
#zwl: val, test计算只有前向，没有后向； 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    model     = RegressionModel(len(feature_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.size(0)
        train_loss = train_loss_sum / len(train_ds)

        # 验证阶段
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze()
                val_loss_sum += criterion(preds, y_batch).item() * X_batch.size(0)
        val_loss = val_loss_sum / len(val_ds)

        if epoch % 10 == 0 or epoch == num_epochs:
            print(f" Epoch {epoch:2d}/{num_epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

    # 测试集评估
    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).squeeze()
            test_loss_sum += criterion(preds, y_batch).item() * X_batch.size(0)
    test_loss = test_loss_sum / len(test_ds)
    print(f"batch_size = {batch_size} | 最终测试 MSE: {test_loss:.4f}")

