"""TAG-ECGNet: Type-Aware Gated ECG Network.

The output is a vector of multilabel logits. Apply sigmoid independently per
category at inference; do not apply softmax over diagnosis categories.
The architecture and state_dict keys match the final research experiment.
"""

import torch
from torch import nn
from torchvision import models


class TAGECGNet(nn.Module):

    def __init__(self, variant="fusion", nlabels=23, nmeta=40, pretrained=True):
        super().__init__()
        self.variant = variant
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone.classifier = nn.Identity()
        self.type_emb = nn.Embedding(2, 32)
        self.attention = nn.Sequential(
            nn.Linear(1280 + 32, 96), nn.Tanh(), nn.Linear(96, 1)
        )
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)
        self.pool_gate = nn.Parameter(torch.full((2,), -2.1972246))
        self.excite = nn.Sequential(
            nn.Linear(1280 + 32, 80), nn.SiLU(), nn.Linear(80, 1280)
        )
        nn.init.zeros_(self.excite[-1].weight)
        nn.init.zeros_(self.excite[-1].bias)
        self.image_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, nlabels))
        if variant == "fusion":
            self.meta_net = nn.Sequential(
                nn.Linear(nmeta, 96), nn.SiLU(), nn.Dropout(0.2), nn.Linear(96, nlabels)
            )
            self.fusion_gate = nn.Linear(1280 + 32, nlabels)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, -2.944439)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def pool(self, h, lengths, typ):
        zs = []
        att = h.new_zeros((len(lengths), 8))
        emb = self.type_emb(typ)
        for i, s in enumerate(h.split(lengths.tolist())):
            n = len(s)
            scores = self.attention(torch.cat([s, emb[i].expand(n, -1)], 1)).squeeze(1)
            w = scores.softmax(0)
            g = self.pool_gate[typ[i]].sigmoid()
            w = (1 - g) / n + g * w
            att[i, :n] = w
            zs.append((w[:, None] * s).sum(0))
        z = torch.stack(zs)
        context = torch.cat([z, emb], 1)
        z = z * (1 + 0.2 * torch.tanh(self.excite(context)))
        return (z, att, emb)

    def forward(self, images, lengths, typ, metadata):
        h = self.backbone(images)
        z, att, emb = self.pool(h, lengths, typ)
        logits = self.image_head(z)
        if self.variant == "fusion":
            available = (metadata[:, metadata.shape[1] // 2 :].sum(1) > 0).to(z.dtype)[
                :, None
            ]
            if self.training:
                available = available * (torch.rand_like(available) > 0.3)
            numeric = self.meta_net(metadata)
            gate = self.fusion_gate(torch.cat([z, emb], 1)).sigmoid()
            logits = logits + available * gate * numeric
        return (logits, att)
