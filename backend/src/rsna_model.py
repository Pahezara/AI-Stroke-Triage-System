#src/rsna_model.py

import torch
import torch.nn as nn
import timm

NUM_CLASSES = 6  


class HemorrhageNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(device="cuda"):
    model = HemorrhageNet()
    model = model.to(device)
    return model


if __name__ == "__main__":
    model = create_model()
    x = torch.randn(2, 3, 512, 512).cuda()
    y = model(x)
    print(y.shape)  #(2, 6)
