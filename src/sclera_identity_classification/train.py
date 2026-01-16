import time
import torch
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from data import make_dataloaders
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config, net, train_loader, val_loader):
    optimizer = instantiate(config.optimizer, params=net.parameters())
    loss_function = nn.CrossEntropyLoss()

    lrs = []
    losses = []
    batch = 1

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
                wandb.log({"batch": batch, "train/batch_loss": loss.item()})
                batch += 1

                # Backward pass
                loss.backward()
                optimizer.step()
                losses.append(loss)

            # validate
            auc = evaluate(net, val_loader, config.model.out_channels)

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": epoch_loss,
                    "validation/epoch_auc": auc.item(),
                }
            )

            print("epoch: {:3d} | lr: {:} | epoch_loss: {:7.5f} | val_auc: {:7.5f} | time: {:.2f}".format(epoch, lrs[-1], losses[-1].item() / config.batch_size, auc.item(), time.time() - start_time))

            if config.saving_period > 0 and config.model_save_path is not None and epoch % config.saving_period == 0:
                torch.save(net, config.model_save_path + "_" + str(epoch) + ".pth")

    except KeyboardInterrupt:
        print("user interrupted training")
    return net, losses


def init_net(config):
    net = instantiate(config.model)
    net.to(device)
    print(f"Total paramters: {sum([p.numel() for p in net.parameters()])}")
    return net


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def main(config):

    print(f"Running on {device}")
    print(f"Running the following exepriment: {config.experiment_name}")


    wandb.init(
        project=config.wandb.project,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # # Defining different steps for the wandb charts so they are not shared
    wandb.define_metric("train/batch_*", step_metric="batch")
    wandb.define_metric("train/epoch_*", step_metric="epoch")
    wandb.define_metric("validation/epoch_*", step_metric="epoch")

    # Prepare the data:
    train_loader, val_loader, test_loader = make_dataloaders(config)

    # Initialize the model/network:
    net = init_net(config)
    # print('Net output: ', net)

    wandb.watch(net, log="gradients", log_freq=100)

    # Train model:
    net, losses = train(config, net=net, train_loader=train_loader, val_loader=val_loader)

    test_auc = evaluate(net, test_loader)
    wandb.run.summary["test/auc"] = float(test_auc.item())

    # Save trained model:
    torch.save(net, config.model_save_path + ".pth")
    artifact = wandb.Artifact(name="sclera-identity-classification-model", type="model", description="A model trained to identify individuals based on sclera images", metadata={"auc": float(test_auc.item())})
    artifact.add_file(config.model_save_path + ".pth")
    wandb.log_artifact(artifact)
    wandb.finish()

    return net, test_loader, losses


if __name__ == "__main__":
    main()
