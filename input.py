import os
import json

import numpy as np

import torch
from torch.utils.data import Dataset


class BabiTaskDataset(Dataset):
    """bAbI task dataset."""

    TRAIN = "train"
    TEST = "test"

    def __init__(self, dataset_folder_path, task_id, jointly_preprocessed, split):
        # Sanity checks
        assert split in (BabiTaskDataset.TRAIN, BabiTaskDataset.TEST)

        # Attributes
        self.dataset_folder_path = dataset_folder_path
        self.task_id = task_id
        self.jointly_preprocessed = jointly_preprocessed
        self.split = split
        self.metadata = None
        self.dataset = None

        self.load_dataset()

    def load_dataset(self):
        task_name = "qa%d" % self.task_id
        path_suffix = "_metadata.json" if not self.jointly_preprocessed else "_joint_metadata.json"
        metadata_path = os.path.join(
            self.dataset_folder_path, task_name + path_suffix)

        with open(metadata_path, "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        dataset_path = os.path.join(
            self.dataset_folder_path, self.metadata["filenames"][self.split])

        with open(dataset_path, "r") as dataset_file:
            self.dataset = json.load(dataset_file)

    def __len__(self):
        return len(self.dataset)
    
    def _get_supporting_mask(self, supporting_idx):
        mask = np.zeros(self.metadata["max_story_length"])
        for idx in supporting_idx:
            mask[idx - 1] = 1
        return mask

    def __getitem__(self, id_):
        raw_sample = self.dataset[id_]
        data_fields = ("story", "query", "answer")
        sample = {}
        for i, field in enumerate(data_fields):
            sample[field] = torch.from_numpy(np.array(raw_sample[i]))
        sample["supporting"] = torch.from_numpy(self._get_supporting_mask(raw_sample[3]))
        sample["story_mask"] = self._get_story_mask(raw_sample[0])
        return sample
    
    def _get_story_mask(self, story):
        mask = np.zeros(self.metadata["max_story_length"])
        for i, fact in enumerate(story):
            if fact[0] != 0:
                mask[i] = 1
        return mask

    def translate_story(self, raw_story, raw_query, raw_answer):
        vocab = self.metadata["vocab"]
        answers_vocab = self.metadata["answers_vocab"]
        # raw_story, raw_query, raw_answer, supporting = self.dataset[id_]
        story = []
        for i, raw_fact in enumerate(raw_story):
            if raw_fact[0] == 0:
                break
            fact = " ".join([vocab[j] for j in raw_fact if j != 0])
            story.append((i + 1, fact))
        query = " ".join([vocab[i] for i in raw_query])
        answer = answers_vocab[raw_answer]
        return story, query, answer