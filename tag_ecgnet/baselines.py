"""Image-only full-backbone CNN references with report mean-logit pooling."""

import torch
from torch import nn
from torchvision import models

ARCHS = {
    "resnet18": ("ResNet18_Weights", "IMAGENET1K_V1"),
    "mobilenet_v2": ("MobileNet_V2_Weights", "IMAGENET1K_V1"),
    "shufflenet_v2_x1_0": ("ShuffleNet_V2_X1_0_Weights", "IMAGENET1K_V1"),
    "squeezenet1_1": ("SqueezeNet1_1_Weights", "IMAGENET1K_V1"),
    "efficientnet_b0": ("EfficientNet_B0_Weights", "IMAGENET1K_V1"),
}


def build(name, nlabels, pretrained=True):
    enum, version = ARCHS[name]
    weights = getattr(getattr(models, enum), version) if pretrained else None
    m = getattr(models, name)(weights=weights)
    if name in ["resnet18", "shufflenet_v2_x1_0"]:
        dim = m.fc.in_features
        m.fc = nn.Linear(dim, nlabels)
    elif name in ["mobilenet_v2", "efficientnet_b0"]:
        dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(dim, nlabels)
    else:
        m.classifier[1] = nn.Conv2d(512, nlabels, kernel_size=1)
        m.classifier[2] = nn.Identity()
        m.num_classes = nlabels
    return m


class Classic(nn.Module):

    def __init__(self, name, nlabels, pretrained=True):
        super().__init__()
        self.backbone = build(name, nlabels, pretrained)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def forward(self, images, lengths):
        logits = self.backbone(images)
        splits = logits.split(lengths.tolist())
        z = torch.stack([s.mean(0) for s in splits])
        att = logits.new_zeros((len(lengths), 8))
        for i, n in enumerate(lengths.tolist()):
            att[i, :n] = 1 / n
        return (z, att)


class ImageOnlyBaseline(Classic):
    def forward(self, images, lengths, typ=None, metadata=None):
        return super().forward(images, lengths)
