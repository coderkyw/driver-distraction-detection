import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import ResNetClassifier
from models.mobilenet_model import MobileNetClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CLASSES = [
    "safe driving", "texting - right", "talking on the phone - right",
    "texting - left", "talking on the phone - left", "operating the radio",
    "drinking", "reaching behind", "hair and makeup", "talking to passenger"
]

def predict_image(image_path, model_choice='resnet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_choice == 'resnet':
        model = ResNetClassifier(num_classes=10).to(device)
        model_path = os.path.join(BASE_DIR, 'resnet_best.pth')
    elif model_choice == 'mobilenet':
        model = MobileNetClassifier(num_classes=10).to(device)
        model_path = os.path.join(BASE_DIR, 'mobilenet_best.pth')
    else:
        raise ValueError("不支持的模型选择，仅支持 'resnet' 或 'mobilenet'")

    print(f"[INFO] 加载模型权重: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    class_name = CLASSES[pred.item()]
    print(f"✅ 模型: {model_choice} | 预测类别: {class_name}")
    return class_name
