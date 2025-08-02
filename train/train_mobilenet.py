"""
训练 MobileNet 模型示例
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

# === 导入 ===
from models.mobilenet_model import MobileNetClassifier
from utils.dataset import get_dataloaders
from utils.visualize import plot_confusion_matrix

print("MobileNetClassifier import success!")

# cudnn 设置（可选）
cudnn.benchmark = False
cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 参数
num_epochs = 10
num_classes = 10
batch_size = 8   # 可以小一点以免显存溢出
learning_rate = 1e-4

# ✅ 用绝对路径确保不报错
data_dir = os.path.join(BASE_DIR, 'data', 'processed', 'train')
print("当前工作目录：", os.getcwd())
print(f"加载训练数据目录：{data_dir}")

# 数据加载
train_loader, val_loader = get_dataloaders(batch_size=batch_size, data_dir=data_dir, num_workers=0)

# 模型
model = MobileNetClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}] 开始训练")
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] 完成，平均 Loss: {avg_loss:.4f}")

    # 验证
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

    plot_confusion_matrix(all_labels, all_preds)

    torch.cuda.empty_cache()

torch.save(model.state_dict(), os.path.join(BASE_DIR, "mobilenet_best.pth"))
print(f"✅ MobileNet 模型已保存到 mobilenet_best.pth")