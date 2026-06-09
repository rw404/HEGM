import os
import sys
import json
import yaml
import torch
import numpy as np
import polars as pl
import argparse
from tqdm import tqdm

from hegm.model import (
    FeatureEmbedder,
    DCNEncoder,
    ResNetDecoderHEGM,
    UnifiedRanker,
)
from hegm.features import compute_bucket_edges, compute_cat_cardinalities
from hegm.hegm import decode_predictions
from hegm.metrics import batch_roc_auc_consumptions


def evaluate_model(ckpt_path: str, config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    paths_cfg = config["paths"]

    num_cont = dataset_cfg["num_cont_features"]
    num_cat = dataset_cfg["num_cat_features"]
    emb_dim = model_cfg["emb_dim"]
    n_gaussians = model_cfg["n_gaussians"]
    dcn_dropout = model_cfg.get("dcn_dropout", 0.1)
    time_scale = dataset_cfg["time_scale"]

    train_path = paths_cfg["train"]
    test_path = paths_cfg["test"]

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
    norm_constant = meta["NORM_CONSTANT"]
    num_users = meta.get("num_users")
    num_items = meta.get("num_items")
    norm_sec = norm_constant / time_scale

    bucket_edges = compute_bucket_edges(train_path, num_cont, border_count=32)
    cat_cardinalities = compute_cat_cardinalities(train_path, num_cat)

    embedder = FeatureEmbedder(
        num_cont_features=num_cont,
        bucket_edges=bucket_edges,
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
    )

    dcn_input_features = num_cont + num_cat
    if num_users is not None:
        dcn_input_features += 2

    flattened_dim = dcn_input_features * emb_dim
    encoder = DCNEncoder(n_features=flattened_dim, num_layers=2, init_dropout_p=dcn_dropout)

    decoder = ResNetDecoderHEGM(flattened_dim, n_gaussians)

    model = UnifiedRanker(
        embedder,
        encoder,
        decoder,
        num_users=num_users,
        num_items=num_items,
        emb_dim=emb_dim,
    )

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please run scripts/download_artifacts.py to download pre-trained models.")
        sys.exit(1)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    df = pl.read_parquet(test_path)

    num_features = torch.tensor(np.vstack(df["num_features"].to_numpy()), dtype=torch.float32)
    cat_features = torch.tensor(np.vstack(df["cat_features"].to_numpy()), dtype=torch.long)
    targets_norm = torch.tensor(df["consumption_time"].to_numpy(), dtype=torch.float32)

    has_ids = num_users is not None
    if has_ids:
        user_ids = torch.tensor(df["user_id"].to_numpy(), dtype=torch.long)
        item_ids = torch.tensor(df["item_id"].to_numpy(), dtype=torch.long)
    else:
        user_ids, item_ids = None, None

    raw_preds_list = []
    batch_size = 4096

    xauc_num_total, xauc_den_total = 0.0, 0.0

    for i in tqdm(range(0, len(num_features), batch_size), desc="Evaluating"):
        with torch.no_grad():
            b_num = num_features[i : i + batch_size].to(device)
            b_cat = cat_features[i : i + batch_size].to(device)
            b_users = user_ids[i : i + batch_size].to(device) if has_ids else None
            b_items = item_ids[i : i + batch_size].to(device) if has_ids else None
            b_targets = targets_norm[i : i + batch_size].to(device)

            raw_preds = model(b_num, b_cat, user_id=b_users, item_id=b_items)
            raw_preds_list.append(raw_preds.cpu())

            exp_vt = decode_predictions(raw_preds)
            corrects, totals = batch_roc_auc_consumptions(b_targets, exp_vt)
            xauc_num_total += corrects.item()
            xauc_den_total += totals.item()

    raw_preds = torch.cat(raw_preds_list)
    expected_viewtime_norm = decode_predictions(raw_preds)

    targets_sec = targets_norm * norm_sec
    expected_viewtime_sec = expected_viewtime_norm * norm_sec

    mae = torch.nn.functional.l1_loss(expected_viewtime_sec, targets_sec).item()
    mse = torch.nn.functional.mse_loss(expected_viewtime_sec, targets_sec).item()

    xauc = xauc_num_total / xauc_den_total if xauc_den_total > 0 else 0.0

    dataset_name = dataset_cfg["name"].upper()
    print(f"\nHEGM on {dataset_name}, K={n_gaussians}")
    print(f"MSE:  {mse:.4f} sec^2")
    print(f"MAE:  {mae:.4f} sec")
    print(f"XAUC: {xauc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HEGM model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    evaluate_model(ckpt_path=args.checkpoint, config=config)
