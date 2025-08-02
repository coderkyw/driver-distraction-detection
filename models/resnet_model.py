
import torch.nn as nn
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
