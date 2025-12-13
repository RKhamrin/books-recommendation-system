from books_recommendation_system.books_recommendation_system.data import BERT4RecDataModule
from books_recommendation_system.books_recommendation_system.models import BERT4RecModel

import torch
import pytorch_lightning as pl

def train_bert4rec(train_df, valid_df, test_df):
    """🚀 Тренирует BERT4Rec с правильным порядком инициализации"""
    
    # Шаг 1: Создаем DataModule
    dm = BERT4RecDataModule(train_df, valid_df, test_df, batch_size=32, max_len=30)
    
    # Шаг 2: Вызываем setup() ДО создания модели
    dm.setup('fit')
    
    # Шаг 3: Создаем модель с уже известным n_items
    model = BERT4RecModel(
        n_items=dm.n_items,
        hidden_dim=64,
        n_layers=2,
        n_heads=1,
        max_len=30,
        lr=5e-4
    )
    
    # Шаг 4: Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ]
    )
    
    # Обучение
    trainer.fit(model, dm)
    trainer.test(model, dm)
    
    return model, trainer, dm