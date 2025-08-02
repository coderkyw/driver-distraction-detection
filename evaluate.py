"""
模型评估脚本
"""

import os
import sys
import torch
import torch.nn as nn
from utils.dataset import get_dataloaders
from models.resnet_model import ResNetClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate(data_dir, model_path="resnet_best.pth", batch_size=32, num_classes=10):
    print(f"[INFO] 加载验证数据目录: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"[ERROR] 验证数据路径不存在: {data_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据，只用验证集
    _, val_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # 加载模型
    model = ResNetClassifier(num_classes=num_classes).to(device)
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型权重文件不存在: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"验证集准确率 (Accuracy): {accuracy:.4f}")
    print(f"验证集 F1-score (Macro): {f1:.4f}")

    classes = [f"c{i}" for i in range(num_classes)]
    plot_confusion_matrix(cm, classes=classes, title='验证集混淆矩阵')

if __name__ == "__main__":
    # 确保项目根目录正确
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
    # 如果你的 evaluate.py 在 train/ 或其他子文件夹，改成往上两级：os.path.dirname(os.path.dirname(...))

    # 假设你的验证集目录结构是 data/processed/val
    val_data_dir = os.path.join(BASE_DIR, "data", "processed", "val")
    model_weights_path = os.path.join(BASE_DIR, "resnet_best.pth")

    print(f"[INFO] 项目根目录: {BASE_DIR}")
    print(f"[INFO] 验证数据目录: {val_data_dir}")
    print(f"[INFO] 模型权重路径: {model_weights_path}")

    evaluate(data_dir=val_data_dir, model_path=model_weights_path)