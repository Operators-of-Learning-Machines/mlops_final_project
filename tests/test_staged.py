import torch
import pytest
from hydra import initialize, compose
from hydra.utils import instantiate

from sclera_identity_classification.data import make_dataloaders
from sclera_identity_classification.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default_config")
    return cfg


@pytest.fixture(scope="session")
def staged_model(config):
    model = instantiate(config.model)
    state = torch.load(config.model_save_path + ".pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture(scope="session")
def test_loader(config):
    _, _, test_loader = make_dataloaders(config)
    return test_loader


def test_staged_model_auc(staged_model, test_loader, config):
    """
    Hard gate:
    - evaluates real test data
    - fails if performance regresses
    """

    with torch.no_grad():
        auc = evaluate(
            staged_model,
            test_loader,
            config.model.out_channels,
        )

    auc_value = float(auc.item())

    # Set this once based on a known-good run
    MIN_ACCEPTABLE_AUC = 0.50

    assert abs(auc_value) >= MIN_ACCEPTABLE_AUC, (
        f"Staged model AUC too low: {auc_value:.4f} "
        f"(expected abs(AUCadd) >= {MIN_ACCEPTABLE_AUC})"
    )
