import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model.contrastive_gin import GINSimclr
from torch_geometric.data import LightningDataset
from data_provider.pretrain_datamodule import GINPretrainDataModule
from data_provider.pretrain_dataset import GINPretrainDataset


def main(args):
    pl.seed_everything(args.seed)

    # data
    # train_dataset = GINPretrainDataset(args.root, args.text_max_len, args.graph_aug1, args.graph_aug2)
    # dm = LightningDataset(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dm = GINPretrainDataModule.from_argparse_args(args)


    # model
    model = GINSimclr(
        temperature=args.temperature,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_num_layers=args.gin_num_layers,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.graph_pooling,
        graph_self=args.graph_self,
        bert_hidden_dim=args.bert_hidden_dim,
        bert_pretrain=args.bert_pretrain,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print('total params:', sum(p.numel() for p in model.parameters()))

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/pretrain_gin/", every_n_epochs=5))
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, strategy=strategy)

    trainer.fit(model, datamodule=dm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--default_root_dir', type=str, default='./checkpoints/', help='location of model checkpoints')
    # parser.add_argument('--max_epochs', type=int, default=500)

    # GPU
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser = Trainer.add_argparse_args(parser)
    parser = GINSimclr.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args
    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    main(args)








