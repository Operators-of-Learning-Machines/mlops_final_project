import torch
from torch import nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights



class SqueezeNet(nn.Module):
    def __init__(self, pretrained=False, transfer_learning_model_path=None, out_channels=220, kernel_size=1):
        super(SqueezeNet, self).__init__()
        if transfer_learning_model_path is not None:
            self.model = torch.load(transfer_learning_model_path, weights_only=False)
        else:
            self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None)

        self.model.classifier[1] = nn.Conv2d(self.model.classifier[1].in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = self.model.classifier(x)
        return torch.flatten(x, 1)
