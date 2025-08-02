"""
数据集工具
"""

import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir='./data/processed/train', batch_size=32, num_workers=2):
    print(f"🚩 [dataset.py] 加载数据目录: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ 数据目录不存在: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"✅ [dataset.py] 训练样本: {train_size}, 验证样本: {val_size}")
    print(f"✅ [dataset.py] Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")

    return train_loader, val_loader