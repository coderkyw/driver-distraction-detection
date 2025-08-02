
import torch.nn as nn
import torchvision.models as models

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)
