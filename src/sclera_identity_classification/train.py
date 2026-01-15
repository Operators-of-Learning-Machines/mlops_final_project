import time
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sclera_dataset import ScleraDataset
from torcheval.metrics import MulticlassAUROC


device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_data(config):

    base_transform = transforms.Compose(
        [
            transforms.Grayscale(config.channels),
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


    dataset = ScleraDataset(csv_file="./data/labels.csv", root_dir="data", transform=base_transform, gaze_direction=config.gaze_direction)

    # Define the sizes for each subset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # Remaining 15% for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset.dataset.transform = aug_transform

    print(f"Total size: {total_size}, Training size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=100)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=100)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=100)

    return train_loader, val_loader, test_loader


def init_net(config):

    # Initialize the network:
    if config.model_load_path is not None:
        try:
            net = torch.load(config.model_load_path)
            return net
        except FileNotFoundError:
            print('File not found')
    else:
        net = instantiate(config.model)
        net.to(device)
        print(f"Total paramters: {sum([p.numel() for p in net.parameters()])}")
        return net



def validate(net, data_loader, classes=220):
    metric = MulticlassAUROC(num_classes=classes, average="macro").to(device)
    net.eval()
    with torch.no_grad():
        for first, target in data_loader:
            first, target = first.to(device), target.to(device)
            first_out = net(first)
            metric.update(first_out, target)
    return metric.compute()


def train(config, net, train_loader, val_loader):

    # Initialize the optimizer:
    optimizer = instantiate(config.optimizer, params=net.parameters())
    loss_function = nn.CrossEntropyLoss()

    lrs = []
    losses = []

    try:
        for epoch in range(config.epochs):
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
            auc = validate(net, val_loader, config.model.out_channels)

            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_loss,
                "val/auc": auc.item(),
                "lr": optimizer.param_groups[0]["lr"],
            })

            print("epoch: {:3d} | lr: {:} | epoch_loss: {:7.5f} | val_auc: {:7.5f} | time: {:.2f}".format(epoch, lrs[-1], losses[-1].item() / config.batch_size, auc.item(), time.time() - start_time))

            if config.saving_period > 0 and config.model_save_path is not None and epoch % config.saving_period == 0:
                torch.save(net, config.model_save_path + "_" + str(epoch) + ".pth")

    except KeyboardInterrupt:
        print("user interrupted training")
    return net, losses



@hydra.main(version_base=None, config_path='../../configs', config_name='default_config')
def main(config):

    print(f"Running on {device}")

    wandb.init(
        project=config.wandb.project,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Prepare the data:
    train_loader, val_loader, test_loader = prepare_data(config)

    # Initialize the model/network:
    net = init_net(config)
    # print('Net output: ', net)

    wandb.watch(net, log="gradients", log_freq=100)

    # Train model:
    net, losses = train(config, net=net, train_loader=train_loader, val_loader=val_loader)

    # Save trained model:
    torch.save(net, config.model_save_path + ".pth")
    wandb.save(config.model_save_path)

    test_auc = validate(net, test_loader)
    print("test val_auc: {:7.5f}".format(test_auc.item()))

    wandb.log({"test/auc": test_auc.item()})
    wandb.finish()

    return net, test_loader, losses


if __name__ == "__main__":
    main()
