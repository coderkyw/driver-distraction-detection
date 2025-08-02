"""
训练 ResNet 模型示例
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score

# === 自动将根目录加入 sys.path ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("DEBUG: sys.path =", sys.path)

# === 这里正式导入 ===
from models.resnet_model import ResNetClassifier
from utils.dataset import get_dataloaders
from utils.visualize import plot_confusion_matrix

print("ResNetClassifier import success!")

# 关闭 cudnn 自动调优，避免显存碎片和卡顿
cudnn.benchmark = False
cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 参数设置
num_epochs = 10
num_classes = 10
batch_size = 8  # 小一点减轻显存压力
learning_rate = 1e-4
data_dir = os.path.join(BASE_DIR, 'data', 'processed', 'train')

print(f"当前工作目录：{os.getcwd()}")
print(f"加载训练数据目录：{data_dir}")

# 加载数据，num_workers=0避免Windows多进程卡顿
train_loader, val_loader = get_dataloaders(batch_size=batch_size, data_dir=data_dir, num_workers=0)

model = ResNetClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("[INFO] 进入主函数入口 🚀")

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}] 开始训练")
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        print(f"[Epoch {epoch+1}] [Batch {i+1}/{len(train_loader)}] 数据加载完成")
        images, labels = images.to(device), labels.to(device)

        print(f"[Epoch {epoch+1}] [Batch {i+1}] 前向传播开始")
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        print(f"[Epoch {epoch+1}] [Batch {i+1}] 反向传播开始")
        loss.backward()

        print(f"[Epoch {epoch+1}] [Batch {i+1}] 优化器更新开始")
        optimizer.step()

        total_loss += loss.item()
        print(f"[Epoch {epoch+1}] [Batch {i+1}] 完成, Loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] 完成，平均 Loss: {avg_loss:.4f}")

    # 验证阶段
    print(f"Epoch [{epoch+1}] 开始验证")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Validation F1-score: {f1:.4f}")

    # 绘制混淆矩阵（会弹图，需要可视化环境）
    plot_confusion_matrix(all_labels, all_preds)

    # 训练完每个 epoch 清理显存
    torch.cuda.empty_cache()

# 保存模型
model_path = os.path.join(BASE_DIR, "resnet_best.pth")
torch.save(model.state_dict(), model_path)
print(f"模型已保存到 {model_path}")