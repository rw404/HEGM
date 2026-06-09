import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class VKLSVDDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pl.read_parquet(parquet_path)

        self.num_features = torch.tensor(df["num_features"].to_list(), dtype=torch.float32)
        self.cat_features = torch.tensor(df["cat_features"].to_list(), dtype=torch.long)
        self.consumption_time = torch.tensor(df["consumption_time"].to_numpy(), dtype=torch.float32)

        if "user_id" in df.columns and "item_id" in df.columns:
            self.user_id = torch.tensor(df["user_id"].to_numpy(), dtype=torch.long)
            self.item_id = torch.tensor(df["item_id"].to_numpy(), dtype=torch.long)
            self.has_ids = True
        else:
            self.has_ids = False

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        batch = {
            "num_features": self.num_features[idx],
            "cat_features": self.cat_features[idx],
            "consumption_time": self.consumption_time[idx],
        }
        if self.has_ids:
            batch["user_id"] = self.user_id[idx]
            batch["item_id"] = self.item_id[idx]
        return batch


class LocalDataModule(L.LightningDataModule):
    def __init__(self, train_path: str, val_path: str, batch_size: int = 1024):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = VKLSVDDataset(self.train_path)
        self.val_ds = VKLSVDDataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
