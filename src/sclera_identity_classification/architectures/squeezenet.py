import torch
from torch import nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

class SqueezeNet(nn.Module):
    def __init__(self, pretrained=False, transfer_learning_model_path=None, out_channels=220, kernel_size=1):
        super().__init__()
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.classifier[1] = nn.Conv2d(
            self.model.classifier[1].in_channels,
            out_channels,
            kernel_size
        )
        if transfer_learning_model_path:
            self.load_state_dict(torch.load(transfer_learning_model_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = self.model.classifier(x)
        return torch.flatten(x, 1)
