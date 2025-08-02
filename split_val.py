"""
split_val.py
把 processed/train 中的图片按类别随机抽 20% 放到 processed/val，做验证集
"""

import os
import shutil
import random

# === 你可以根据自己路径修改这两个 ===
train_dir = './data/processed/train'
val_dir = './data/processed/val'
val_ratio = 0.2  # 验证集占比

# === 创建 val 目录 ===
os.makedirs(val_dir, exist_ok=True)

# === 按类别处理 ===
for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    if not os.path.isdir(class_train_dir):
        continue

    os.makedirs(class_val_dir, exist_ok=True)

    images = os.listdir(class_train_dir)
    val_count = int(len(images) * val_ratio)
    val_images = random.sample(images, val_count)

    for img in val_images:
        src = os.path.join(class_train_dir, img)
        dst = os.path.join(class_val_dir, img)
        shutil.move(src, dst)

    print(f"✅ [{class_name}] Train: {len(images)-val_count} Val: {val_count}")

print("🎉 验证集拆分完成！")