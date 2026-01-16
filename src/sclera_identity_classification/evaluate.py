import torch
from torcheval.metrics import MulticlassAUROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(net, data_loader, classes=220):
    metric = MulticlassAUROC(num_classes=classes, average="macro").to(device)
    net.eval()
    with torch.no_grad():
        for first, target in data_loader:
            first, target = first.to(device), target.to(device)
            first_out = net(first)
            metric.update(first_out, target)
    return metric.compute()
