import torch
import numpy as np
import polars as pl


def compute_bucket_edges(parquet_path: str, num_features: int, border_count: int = 32):
    df = pl.read_parquet(parquet_path)
    features = torch.tensor(np.vstack(df["num_features"].to_numpy()), dtype=torch.float32)

    edges = torch.zeros(border_count, num_features)
    quantiles = torch.linspace(0, 1, border_count)

    for i in range(num_features):
        edges[:, i] = torch.quantile(features[:, i], quantiles)
    return edges


def compute_cat_cardinalities(parquet_path: str, num_cat_features: int):
    if num_cat_features == 0:
        return []
    df = pl.read_parquet(parquet_path)
    cat_array = np.vstack(df["cat_features"].to_numpy())
    cardinalities = (cat_array.max(axis=0) + 1).tolist()
    return cardinalities
