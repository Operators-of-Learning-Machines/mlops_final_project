import os
import pytest
from sclera_identity_classification.data import download_and_extract, ScleraDataset
from tests import _PATH_DATA
from torchvision import transforms


@pytest.fixture(scope="session", autouse=True)
def load_data():
    if not os.path.exists(_PATH_DATA):
        print("Data not found. Downloading...")
        download_and_extract()


@pytest.fixture(scope="session", autouse=True)
def init_dataset(load_data):
    base_transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    dataset = ScleraDataset(
        csv_file=os.path.join(_PATH_DATA, "labels.csv"),
        root_dir=_PATH_DATA,
        transform=base_transform
    )
    return dataset


def test_data_path(load_data):
    _ = load_data
    assert os.path.exists(_PATH_DATA), f"Data path does not exist: {_PATH_DATA}"


def test_item_dimensions(init_dataset):
    dataset = init_dataset
    sample = dataset.get_single_item(0)
    image = sample[0]
    actual_shape = list(image.shape)
    expected_shape = [3, 400, 400]
    assert actual_shape == expected_shape, f"Unexpected image shape: {actual_shape}"


def test_positive_items_exist(init_dataset):
    dataset = init_dataset
    for i in range(220):
        try:
            item = dataset.get_positive_item(i)
        except IndexError:
            pytest.fail(f"ID {i} not in dataset (IndexError)")
        assert item is not None, f"ID {i} returned None"
