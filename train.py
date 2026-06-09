import os
import sys
import json
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse

from hegm.model import (
    FeatureEmbedder,
    DCNEncoder,
    ResNetDecoderHEGM,
    UnifiedRanker,
)
from hegm.data import LocalDataModule
from hegm.features import compute_bucket_edges, compute_cat_cardinalities
from hegm.hegm import hegm_loss
from hegm.metrics import batch_roc_auc_consumptions


class BaseTrainer(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_correct_pairs = 0.0
        self.val_total_pairs = 0.0
        self.final_val_auc = 0.0
        self.final_val_mae = 0.0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_validation_epoch_start(self):
        self.val_correct_pairs = 0.0
        self.val_total_pairs = 0.0

    def on_validation_epoch_end(self):
        if self.val_total_pairs > 0:
            self.final_val_auc = self.val_correct_pairs / self.val_total_pairs


class HEGMTrainer(BaseTrainer):
    def _compute_hegm(self, batch):
        y = batch["consumption_time"]
        u_id = batch.get("user_id", None)
        i_id = batch.get("item_id", None)
        scores = self.model(
            batch["num_features"], batch["cat_features"], user_id=u_id, item_id=i_id
        )
        return hegm_loss(scores, y)

    def training_step(self, batch, batch_idx):
        loss, nll, mae, _ = self._compute_hegm(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/nll", nll, on_step=False, on_epoch=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, nll, mae, preds = self._compute_hegm(batch)
        self.final_val_mae = mae

        y = batch["consumption_time"]
        corrects, totals = batch_roc_auc_consumptions(y, preds)
        self.val_correct_pairs += corrects.item()
        self.val_total_pairs += totals.item()


def main(config_path: str):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    L.seed_everything(42)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    paths_cfg = config["paths"]

    num_cont_features = dataset_cfg["num_cont_features"]
    num_cat_features = dataset_cfg["num_cat_features"]
    dataset_name = dataset_cfg["name"]

    emb_dim = model_cfg["emb_dim"]
    n_gaussians = model_cfg["n_gaussians"]
    dcn_dropout = model_cfg.get("dcn_dropout", 0.1)

    batch_size = training_cfg["batch_size"]
    max_epochs = training_cfg["max_epochs"]
    lr = training_cfg["lr"]

    train_path = paths_cfg["train"]
    val_path = paths_cfg["val"]

    meta_path = os.path.join(os.path.dirname(train_path), "meta.json")
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: meta.json not found in {os.path.dirname(train_path)}")
        print(
            "Please ensure you have downloaded the processed datasets or run the preprocessing scripts."
        )
        sys.exit(1)
    num_users = meta.get("num_users")
    num_items = meta.get("num_items")

    cat_cardinalities = compute_cat_cardinalities(train_path, num_cat_features)
    bucket_edges = compute_bucket_edges(train_path, num_cont_features, border_count=32)

    embedder = FeatureEmbedder(
        num_cont_features=num_cont_features,
        bucket_edges=bucket_edges,
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
    )

    dcn_features = num_cont_features + num_cat_features
    if num_users and num_items:
        dcn_features += 2

    flattened_dim = dcn_features * emb_dim

    encoder = DCNEncoder(n_features=flattened_dim, num_layers=2, init_dropout_p=dcn_dropout)

    ranker = UnifiedRanker(
        embedder, encoder, None, num_users=num_users, num_items=num_items, emb_dim=emb_dim
    )

    ranker.decoder = ResNetDecoderHEGM(flattened_dim, n_gaussians)
    lightning_model = HEGMTrainer(model=ranker, lr=lr)

    datamodule = LocalDataModule(train_path, val_path, batch_size)

    ckpt_dir = os.path.join("checkpoints", f"{dataset_name}_k{n_gaussians}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=False,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(model=lightning_model, datamodule=datamodule)

    print(f"\nTraining complete: {dataset_name} (K={n_gaussians})")
    print(f"Val XAUC: {lightning_model.final_val_auc:.4f}")
    print(f"Val MAE (norm): {lightning_model.final_val_mae:.4f}")
    print(f"Checkpoint: {ckpt_dir}/last.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HEGM model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    main(config_path=args.config)
