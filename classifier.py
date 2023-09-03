import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, classes=16):
        self.NUM_CLASSES = classes
        super(Classifier, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.second = nn.Sequential(
            nn.Linear(16*26*26, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, self.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.first(x)
        x = x.view(x.size()[0], 16*26*26)
        x = self.second(x)
        return x
