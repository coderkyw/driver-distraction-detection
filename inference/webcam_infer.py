"""
摄像头实时推理示例
"""

import os
import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.resnet_model import ResNetClassifier

# === 保证可以导入 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# === 设备 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型 ===
model = ResNetClassifier(num_classes=10).to(device)
model_path = os.path.join(BASE_DIR, "resnet_best.pth")
print(f"[INFO] 加载权重: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 预处理 ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 类别标签（你可以改成自己实际的标签） ===
CLASSES = [
    "safe driving", "texting - right", "talking on the phone - right",
    "texting - left", "talking on the phone - left", "operating the radio",
    "drinking", "reaching behind", "hair and makeup", "talking to passenger"
]

# === 打开摄像头 ===
cap = cv2.VideoCapture(0)  # 默认摄像头

if not cap.isOpened():
    print("[ERROR] 无法打开摄像头")
    exit()

print("[INFO] 正在从摄像头读取... 按 Q 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 无法读取帧")
        break

    # === 图像预处理 ===
    input_tensor = transform(frame).unsqueeze(0).to(device)
    # === 推理 ===
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        _, pred = torch.max(probs, 1)
        pred_class = CLASSES[pred.item()]
        confidence = probs[0][pred.item()].item()

    # === 在画面上显示结果 ===
    text = f"{pred_class} ({confidence:.2f})"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)
    cv2.imshow("Driver Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
