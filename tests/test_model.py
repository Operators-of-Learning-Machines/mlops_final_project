import torch
import pytest
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sclera_identity_classification.train import train, init_net
from omegaconf import OmegaConf


@pytest.fixture
def make_dummy_config():
    cfg = OmegaConf.create(
        {
            "epochs": 1,
            "batch_size": 2,
            # static fields from default_config
            "channels": 3,
            "pretrained": False,
            "gaze_direction": "a",
            "saving_period": -1,
            "model_save_path": None,
            "model_load_path": None,
            # model config (required by instantiate)
            "model": {
                "_target_": "sclera_identity_classification.architectures.dummynet.DummyNet",
                "out_channels": 3,
                "in_channels": 3,
            },
            # optimizer config (required by instantiate)
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.01,
            },
            # wandb stub
            "wandb": {
                "project": "test",
                "mode": "disabled",
            },
        }
    )

    return cfg


@pytest.fixture
def dummy_dataloaders():
    batch_size = 4
    num_samples = 8
    in_channels = 3
    height = width = 2
    num_classes = 3

    x = torch.randn(num_samples, in_channels, height, width)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, loader


def test_init_net(make_dummy_config):
    net = init_net(make_dummy_config)

    assert isinstance(net, nn.Module)
    assert sum(p.numel() for p in net.parameters()) > 0


def test_train_runs(make_dummy_config, dummy_dataloaders):

    train_loader, val_loader = dummy_dataloaders
    net = init_net(make_dummy_config)

    wandb.init(
        project=make_dummy_config.wandb.project,
        mode=make_dummy_config.wandb.mode,
        config=OmegaConf.to_container(make_dummy_config, resolve=True),
    )

    trained_net, losses = train(
        make_dummy_config,
        net,
        train_loader,
        val_loader,
    )

    assert isinstance(trained_net, nn.Module)
    assert len(losses) > 0
    assert all(torch.is_tensor(ls) for ls in losses)
