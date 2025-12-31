import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


# 2. BERT4Rec модель (без изменений)
class BERT4RecModel(pl.LightningModule):
    def __init__(
        self, n_items, hidden_dim=128, n_layers=2, n_heads=4, max_len=50, dropout=0.1, lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_items),
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        item_emb = self.item_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(seq_len, device=self.device))
        x = item_emb + pos_emb
        x = self.transformer(x)
        logits = self.mlp(x)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self(input_ids)
        logits_flat = logits.view(-1, self.n_items)
        labels_flat = labels.view(-1)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self(input_ids)
        logits_flat = logits.view(-1, self.n_items)
        labels_flat = labels.view(-1)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]
