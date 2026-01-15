from math import comb
import random
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


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

        # build the dictionary
        self.label_dict = {}
        counter = 0

        for i in range(self.frame.shape[0] - 1):
            index = int(pd.to_numeric(self.frame.iloc[i, 1], errors="coerce"))
            if index not in self.label_dict:
                self.label_dict[index] = counter
                counter += 1


    def __len__(self):
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
        except IndexError as e:
            return self.__getitem__(random.randint(0, self.dataset_length - 1))
