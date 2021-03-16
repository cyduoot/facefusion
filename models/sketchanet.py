import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

model_paths ={
    'sketchanet': './models/sketch_a_net_model_best.pth.tar',
}

class SketchANet(nn.Module):
    def __init__(self, num_classes=250):
        super(SketchANet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=15, stride=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

def sketchanet(pretrained=False, **kwargs):
    model = SketchANet(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_paths['sketchanet']))

    new_classifer = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifer
    return model

class SketchANetModel(nn.Module):
    def __init__(self, num_classes=None):
        super(SketchANetModel, self).__init__()
        self.base = sketchanet(pretrained=False)

        planes = 256
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal(self.fc.weight, std=0.001)
            init.constant(self.fc.bias, 0.1)

    def forward(self, x):
        feat = self.base(x)

        if hasattr(self, 'fc'):
            logits = self.fc(feat)
            return feat, logits

        return feat

def sketchanetmodel(pretrained=False, **kwargs):
    model = SketchANetModel(**kwargs, num_classes=250)
    if pretrained:
        model.load_state_dict(torch.load(model_paths['sketchanet'])['state_dict'])
    return model