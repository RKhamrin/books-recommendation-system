from books_recommendation_system.books_recommendation_system.data.data_module import (
    BERT4RecDataModule,
)
from books_recommendation_system.books_recommendation_system.data.prepare_data import (
    prepare_bert4rec_data,
)
from books_recommendation_system.books_recommendation_system.models import BERT4RecModel

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import pandas as pd


@hydra.main(version_base=None, config_path="../configs", config_name="main_conf")
def train_bert4rec(cfg: DictConfig):
    data = pd.read_csv("../data/ratings.csv")
    item_encoder, sequences, train_df, valid_df, test_df = prepare_bert4rec_data(
        users_items_ratings=data, min_seq_len=cfg.data.min_seq_len
    )
    dm = BERT4RecDataModule(
        train_df, valid_df, test_df, batch_size=cfg.data.batch_size, max_len=cfg.data.max_len
    )
    dm.setup("fit")

    model = BERT4RecModel(
        n_items=dm.n_items,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        max_len=cfg.model.max_len,
        lr=cfg.model.lr,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        ],
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

    return model, trainer, dm
