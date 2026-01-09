import argparse
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, DataLoader
from torchvision import transforms
from squeezenet import SqueezeNet
from sclera_dataset import ScleraDataset
from torcheval.metrics import MulticlassAUROC
import os
import wandb


def train(net, train_loader, val_loader, device, args):

    optimizer = optim.RMSprop(net.parameters(), lr=1e-5, weight_decay=1e-3, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    lrs = []
    losses = []

    try:
        for epoch in range(args.epochs):
            start_time = time.time()
            net.train()
            epoch_loss = 0
            lrs.append(optimizer.param_groups[0]["lr"])

            for first, target in train_loader:
                first, target = first.to(device), target.to(device)

                optimizer.zero_grad()

                out = net(first)

                # Compute loss
                loss = loss_function(out, target)
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                losses.append(loss)

            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch,
            })


            # validate
            auc = validate(net, val_loader)
            print("epoch: {:3d} | lr: {:} | epoch_loss: {:7.5f} | val_auc: {:7.5f} | time: {:.2f}".format(epoch, lrs[-1], losses[-1].item() / args.batch_size, auc.item(), time.time() - start_time))

            wandb.log({
                "val/auc": auc,
                "epoch": epoch,
            })


            if args.saving_period > 0 and args.model_save_path is not None and epoch % args.saving_period == 0:
                torch.save(net, args.model_save_path + "_" + str(epoch) + ".pth")

    except KeyboardInterrupt:
        print("user interrupted training")
    return net, losses


def validate(net, data_loader, classes=220, device="cuda"):
    metric = MulticlassAUROC(num_classes=classes, average="macro").to(device)
    net.eval()
    with torch.no_grad():
        for first, target in data_loader:
            first, target = first.to(device), target.to(device)
            first_out = net(first)
            metric.update(first_out, target)
    return metric.compute()


def main():        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=25, help="number of epochs to train")
    parser.add_argument("-sm", "--model_save_path", type=str, default='models/model', help="path to a save model file")
    parser.add_argument("-lm", "--model_load_path", type=str, default=None, help="path to a pre-trained model file")
    parser.add_argument("-p", "--pretrained", action="store_true", default=False, help="start training with pretrained squeezenet model")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-g", "--gaze_direction", type=str, default="a", help="gaze direction to filter the dataset by")
    parser.add_argument("-t", "--transfer_learning_model_path", type=str, default=None, help="path to a pre-trained model file")
    parser.add_argument("-sp", "--saving_period", type=int, default=-1, help="number of epochs between saving models")
    parser.add_argument("-nt", "--no_train", action="store_true", default=False, help="do not train")
    parser.add_argument("-c", "--channels", type=int, default=3, help="number of channels in the input image")
    parser.add_argument("-a", "--architecture", type=str, default="squeezenet", help="architecture to use for training")

    args = parser.parse_args()


    wandb.init(
        project="sclera-identity-classification",
        config={
            "architecture": args.architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "device": device,
            "pretrained": args.pretrained,
            "channels": args.channels,
        },
    )

    # print the args nicely
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("device:", device, flush=True)

    base_transform = transforms.Compose(
        [
            transforms.Grayscale(args.channels),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    aug_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=(3, 3), translate=(0.1, 0.1), scale=(1.2, 1.2), shear=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.05),
            base_transform,
        ]
    )

    args = parser.parse_args()


    dataset = ScleraDataset(csv_file="data/labels.csv", root_dir="data", transform=base_transform, gaze_direction=args.gaze_direction)
    # Define the sizes for each subset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # Remaining 15% for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset.dataset.transform = aug_transform

    print(f"Total size: {total_size}, Training size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=100)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=100)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=100)

    if args.model_load_path is not None:
        try:
            net = torch.load(args.model_load_path)
        except FileNotFoundError:
            print("File not found")
    else:
        net = SqueezeNet(pretrained=args.pretrained, transfer_learning_model_path=args.transfer_learning_model_path, classes=220)
        net.to(device)
        print(f"Total paramters: {sum([p.numel() for p in net.parameters()])}")

    wandb.watch(net, log="gradients", log_freq=100)

    train(net, train_loader=train_loader, val_loader=val_loader, device=device, args=args)

    torch.save(net, args.model_save_path + ".pth")

    wandb.save(args.model_save_path)

    test_auc = validate(net, test_loader, device=device)

    print("test val_auc: {:7.5f}".format(test_auc.item()))
    wandb.finish()


if __name__ == "__main__":
    main()
