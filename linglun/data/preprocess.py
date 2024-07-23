import os
from pathlib import Path
import random
import json
from random import randint
import numpy as np
import torch


class DataCollector:
    def __init__(
        self,
        data_path,
        seq_length,
        batch_size=8,
        factor=0.2,
        train_ratio=0.8,
        test_ratio=0.2,
    ):
        self.data_path = data_path
        self.files = [file for file in os.listdir(data_path) if file.endswith(".json")]
        self.n_files = len(self.files)
        self._n_train = int((train_ratio - 0.1) * self.n_files)
        self._n_val = int(0.1 * self.n_files)
        self._n_test = int(test_ratio * self.n_files)
        self.datasets = {
            "train": self.files[: self._n_train],
            "val": self.files[self._n_train : self._n_train + self._n_val],
            "test": self.files[self._n_train + self._n_val :],
        }
        self.seq_len = seq_length
        self.factor = factor
        self.batch_size = batch_size

    def _get_seq(self, fname):
        with open(Path(self.data_path, fname), "r") as f:
            seq = json.load(f)
        return np.array(seq["ids"][0])

    def sample_seq(self, seqs, seq_len, factor):
        samples = []
        for seq in seqs:
            lth = min(
                len(seq), seq_len
            )  # get the length of sample sequence, if the sequence is shorter than seq_len, use the length of the sequence
            lth = randint(
                max(0, lth * (1 - factor)), lth * (1 + factor)
            )  # add some randomness to the length of the sample sequence
            start_idx = randint(
                0, len(seq) - lth
            )  # get the starting index of the sample sequence
            samples.append(torch.LongTensor(seq[start_idx : start_idx + lth]))
        return samples

    def get_slide_seq2seq(
        self, seq_len=None, batch_size=None, factor=None, pad_token=0
    ):
        dataset = self.datasets["train"]
        # slide the sequence to make it into seq2seq format
        seq_len = seq_len or self.seq_len
        batch_size = batch_size or self.batch_size
        factor = factor or self.factor
        batch_files = self.batch(dataset, batch_size)
        seqs = []
        for file in batch_files:
            seqs.append(self._get_seq(file))
        samples = self.sample_seq(seqs, seq_len, factor)
        pad_data = torch.nn.utils.rnn.pad_sequence(
            samples, padding_value=pad_token
        ).transpose(-1, -2)
        x = pad_data[:, :-1]
        y = pad_data[:, 1:]
        return x, y, samples

    def batch(self, dataset, batch_size=None):
        batch = dataset.copy()
        batch_size = batch_size or self.batch_size
        np.random.shuffle(batch)
        return batch[:batch_size]
