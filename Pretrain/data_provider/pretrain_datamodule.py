# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
from data_provider.pretrain_dataset import GINPretrainDataset



class GINPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        graph_aug1: str = 'dnodes',
        graph_aug2: str = 'subgraph',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = GINPretrainDataset(root, text_max_len, graph_aug1, graph_aug2)

    def setup(self, stage: str = None):
        self.train_dataset = self.dataset

    def train_dataloader(self):
        loader = torch_geometric.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            # persistent_workers = True
        )
        print('len(train_dataloader)', len(loader))
        return loader