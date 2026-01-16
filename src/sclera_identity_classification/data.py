import os
import random
import zipfile
from typing import Literal

import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

def download_and_extract():
    url = "https://drive.usercontent.google.com/download?id=1H1bS5HXKLVv2WohhP9sqBfbqP0f4AXr5&export=download&authuser=0&confirm=t&uuid=7fac1154-8b63-4e86-b77a-c1b1bcf94517&at=ANTm3cw7hk3aCuEPArnuwUssC7J9%3A1768479815612"
    local_filename = "downloaded_file.zip"
    data_folder = "."

    # Create data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)

    # Streaming download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(local_filename, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("Download complete. Extracting...")

    # Unzip into data folder
    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        zip_ref.extractall(data_folder)

    print(f"Extraction complete. Files are in '{data_folder}/'")

    # Clean up the zip file
    os.remove(local_filename)

class ScleraDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode: Literal["single", "contrastive", "triplet"] = "single", gaze_direction: Literal["s", "l", "r", "u", "a"] = "a"):
        """
        Initialize the ScleraDataset.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing the dataset information.
        root_dir : str
            Path to the root directory containing the images.
        transform : callable, optional
            Optional transform to be applied on a sample.
        mode : Literal["single", "contrastive", "triplet"], optional
            The mode to load the dataset in. Default is "single".
        gaze_direction : Literal["s", "l", "r", "u", "a"], optional
            The gaze direction to filter the dataset by. Default is "a" (all).

        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.gaze_direction = gaze_direction
        self.mode = mode
        new_frame = self.frame.copy()
        self.grouped_by_label_items_dict = dict()
        for i in range(self.frame.shape[0] - 1):
            index = int(pd.to_numeric(self.frame.iloc[i, 1], errors="coerce"))

            if self.gaze_direction != "a":
                cur_gaze_dir = self.frame.iloc[i, 2]
                if cur_gaze_dir != self.gaze_direction:
                    new_frame = new_frame.drop(i)
                    continue
            if index in self.grouped_by_label_items_dict:
                self.grouped_by_label_items_dict[index].append(os.path.join(self.root_dir, str(self.frame.iloc[i, 0])))
            else:
                self.grouped_by_label_items_dict[index] = [os.path.join(self.root_dir, str(self.frame.iloc[i, 0]))]
        self.frame = new_frame

        self.dataset_length = sum(len(items) for items in self.grouped_by_label_items_dict.values())
        # if self.mode == "triplet" or self.mode == "contrastive":
        #     for i in self.grouped_by_label_items_dict:
        #         for j in range(comb(len(self.grouped_by_label_items_dict[j]), 2)):
        #             self.frame.add()

        # build the dictionary
        self.label_dict = {}
        counter = 0

        for i in range(self.frame.shape[0] - 1):
            index = int(pd.to_numeric(self.frame.iloc[i, 1], errors="coerce"))
            if index not in self.label_dict:
                self.label_dict[index] = counter
                counter += 1

    def __len__(self):
        # return self.dataset_length
        return self.frame.shape[0]

    def get_single_item(self, id):

        img_name = os.path.join(self.root_dir, str(self.frame.iloc[id, 0]))
        image = Image.open(img_name).convert("RGB")
        index = int(pd.to_numeric(self.frame.iloc[id, 1], errors="coerce"))
        label = torch.tensor(index, dtype=torch.long)
        # print("single", img_name)
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def get_positive_item(self, anchor_id):
        img_name_candidates = [x for x in self.grouped_by_label_items_dict[anchor_id]]

        if not img_name_candidates:  # Check if the list is empty
            raise IndexError("No valid candidates found for positive sample.")
        img_name = random.choice(img_name_candidates)
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # print("postive", img_name)
        # self.grouped_by_label_items_dict[anchor_id].remove(img_name)

        return image

    def get_negative_item(self, anchor_id):
        different_indices = [x for x in self.grouped_by_label_items_dict.keys() if x != anchor_id]

        if not different_indices:  # Check if there are no different indices
            raise IndexError("No different label groups found.")

        different_index = random.choice(different_indices)
        img_name_candidates = self.grouped_by_label_items_dict[different_index]

        if not img_name_candidates:  # Check if the candidates list is empty
            raise IndexError("No valid candidates found for negative sample.")
        img_name = random.choice(img_name_candidates)
        # print("negative", img_name)
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(
        self,
        idx,
    ):
        try:
            if self.gaze_direction != "a" and str(self.frame.iloc[idx, 2]) != self.gaze_direction:
                raise IndexError("Invalid gaze direction")
            image1, label, index = self.get_single_item(idx)
            if self.mode == "single":
                return image1, label
            elif self.mode == "triplet":
                image2 = self.get_positive_item(index)
                image3 = self.get_negative_item(index)
                return image1, image2, image3, label
            else:
                label = 1 if random.random() > 0.5 else -1
                image2 = self.get_positive_item(index) if label == 1 else self.get_negative_item(index)
                return image1, image2, torch.tensor(label, dtype=torch.long)
        except IndexError:
            return self.__getitem__(random.randint(0, self.dataset_length - 1))


def make_dataloaders(config):

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


if __name__ == "__main__":
    download_and_extract()

    base_transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    dataset = ScleraDataset(csv_file="./data/labels.csv", root_dir="data", transform=base_transform)

    image, label = dataset[0]
    print(f"Single item - Image shape: {list(image.shape)}, Label: {label}")
    print(f"Dataset length: {len(dataset)}")
