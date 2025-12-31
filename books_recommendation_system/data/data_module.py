import pandas as pd
import numpy as np

import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


# 1. Dataset для BERT4Rec (masked item prediction)
class BERT4RecDataset(Dataset):
    def __init__(self, sequences, max_len=50, mask_prob=0.15, n_items=1):
        self.sequences = sequences
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.n_items = n_items
        self.mask_token = n_items

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][: self.max_len]
        seq = seq + [0] * (self.max_len - len(seq))

        input_seq = seq.copy()
        labels = [-100] * self.max_len

        for i in range(len(seq)):
            if np.random.rand() < self.mask_prob and seq[i] != 0:
                input_seq[i] = self.mask_token
                labels[i] = seq[i]

        return {
            "input_ids": torch.tensor(input_seq, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class BERT4RecDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, test_df, batch_size=32, max_len=50, mask_prob=0.15):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.n_items = None
        self.sequences = {}

    def setup(self):
        for split, df in [
            ("train", self.train_df),
            ("valid", self.valid_df),
            ("test", self.test_df),
        ]:
            user_seqs = defaultdict(list)
            for _, row in df.iterrows():
                user_seqs[int(row["user_id"])].append(int(row["item_id"]))
            self.sequences[split] = [seq for seq in user_seqs.values()]

        all_items = set()
        for seqs in self.sequences.values():
            for seq in seqs:
                all_items.update(seq)
        self.n_items = max(all_items) + 1 if all_items else 1

    def train_dataloader(self):
        return DataLoader(
            BERT4RecDataset(self.sequences["train"], self.max_len, self.mask_prob, self.n_items),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            BERT4RecDataset(self.sequences["valid"], self.max_len, self.mask_prob, self.n_items),
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
        )
