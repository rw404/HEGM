import os
import argparse
import polars as pl
import polars.selectors as cs
import numpy as np
import implicit
from scipy.sparse import csr_matrix
from huggingface_hub import hf_hub_download
import json

SUBSAMPLE = "up0.01_ir0.01"
HISTORY_WEEKS = list(range(0, 20))
TRAIN_WEEKS = list(range(20, 25))
VAL_WEEKS = [25]
TEST_WEEKS = [26]
IALS_FACTORS = 32
ALPHA_SMOOTH = 5.0
BETA_SMOOTH = 100.0


def download_data(raw_dir: str):
    files = [f"subsamples/{SUBSAMPLE}/train/week_{i:02}.parquet" for i in range(25)]
    files += [f"subsamples/{SUBSAMPLE}/validation/week_25.parquet"]
    files += [f"subsamples/{SUBSAMPLE}/test/week_26.parquet"]
    files += ["metadata/users_metadata.parquet", "metadata/items_metadata.parquet"]

    for file in files:
        hf_hub_download(
            repo_id="deepvk/VK-LSVD",
            repo_type="dataset",
            filename=file,
            local_dir=raw_dir,
        )


def train_ials(df_history: pl.DataFrame, target_col: str):
    unique_users = df_history["user_id"].unique().to_list()
    unique_items = df_history["item_id"].unique().to_list()

    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {i: j for j, i in enumerate(unique_items)}

    row_idx = (
        df_history["user_id"].map_elements(lambda x: user_map[x], return_dtype=pl.Int32).to_numpy()
    )
    col_idx = (
        df_history["item_id"].map_elements(lambda x: item_map[x], return_dtype=pl.Int32).to_numpy()
    )
    values = df_history[target_col].to_numpy().astype(np.float32)

    sparse_user_item = csr_matrix(
        (values, (row_idx, col_idx)), shape=(len(unique_users), len(unique_items))
    )
    model = implicit.als.AlternatingLeastSquares(
        factors=IALS_FACTORS, iterations=15, regularization=0.01, random_state=42
    )
    model.fit(sparse_user_item, show_progress=True)

    user_factors = model.user_factors
    item_factors = model.item_factors
    user_norms = np.linalg.norm(user_factors, axis=1)
    item_norms = np.linalg.norm(item_factors, axis=1)

    return user_factors, item_factors, user_norms, item_norms, user_map, item_map


def apply_ials(
    df: pl.DataFrame,
    user_factors,
    item_factors,
    user_norms,
    item_norms,
    user_map,
    item_map,
    prefix: str,
):
    u_idx = (
        df["user_id"].map_elements(lambda x: user_map.get(x, -1), return_dtype=pl.Int32).to_numpy()
    )
    i_idx = (
        df["item_id"].map_elements(lambda x: item_map.get(x, -1), return_dtype=pl.Int32).to_numpy()
    )

    valid_mask = (u_idx >= 0) & (i_idx >= 0)

    dot_product = np.zeros(len(df), dtype=np.float32)
    cosine_sim = np.zeros(len(df), dtype=np.float32)
    u_norm_col = np.zeros(len(df), dtype=np.float32)
    i_norm_col = np.zeros(len(df), dtype=np.float32)

    u_f = user_factors[u_idx[valid_mask]]
    i_f = item_factors[i_idx[valid_mask]]

    dot_product[valid_mask] = np.sum(u_f * i_f, axis=1)
    u_norm_col[valid_mask] = user_norms[u_idx[valid_mask]]
    i_norm_col[valid_mask] = item_norms[i_idx[valid_mask]]

    denom = u_norm_col[valid_mask] * i_norm_col[valid_mask]
    cosine_sim[valid_mask] = np.where(denom > 0, dot_product[valid_mask] / denom, 0)

    return df.with_columns(
        [
            pl.Series(f"ials.{prefix}.dot", dot_product),
            pl.Series(f"ials.{prefix}.cos", cosine_sim),
            pl.Series(f"ials.{prefix}.user_norm", u_norm_col),
            pl.Series(f"ials.{prefix}.item_norm", i_norm_col),
        ]
    )


def main(raw_dir: str, out_dir: str):
    download_data(raw_dir)

    items_meta = pl.read_parquet(f"{raw_dir}/metadata/items_metadata.parquet")
    NORM_CONSTANT = items_meta.select(pl.col("duration").quantile(0.99)).item()

    final_out_dir = f"{out_dir}/{SUBSAMPLE}"
    os.makedirs(final_out_dir, exist_ok=True)

    meta_info = {
        "NORM_CONSTANT": float(NORM_CONSTANT),
        "num_users": None,
        "num_items": None,
    }
    with open(f"{final_out_dir}/meta.json", "w") as f:
        json.dump(meta_info, f)

    history_files = [
        f"{raw_dir}/subsamples/{SUBSAMPLE}/train/week_{i:02}.parquet" for i in HISTORY_WEEKS
    ]
    df_history = pl.concat([pl.read_parquet(f) for f in history_files])

    train_files = [
        f"{raw_dir}/subsamples/{SUBSAMPLE}/train/week_{i:02}.parquet" for i in TRAIN_WEEKS
    ]
    df_train = pl.concat([pl.read_parquet(f) for f in train_files])
    df_val = pl.read_parquet(f"{raw_dir}/subsamples/{SUBSAMPLE}/validation/week_25.parquet")
    df_test = pl.read_parquet(f"{raw_dir}/subsamples/{SUBSAMPLE}/test/week_26.parquet")

    df_history = df_history.with_columns(
        [
            (
                pl.col("like")
                | pl.col("share")
                | pl.col("bookmark")
                | pl.col("click_on_author")
                | pl.col("open_comments")
            )
            .cast(pl.Float32)
            .alias("any_feedback"),
            (pl.col("timespent") >= 15).cast(pl.Float32).alias("survival_15"),
            (pl.col("timespent") >= 30).cast(pl.Float32).alias("survival_30"),
            pl.col("timespent").cast(pl.Float32).alias("viewtime_float"),
        ]
    )

    user_stats = df_history.group_by("user_id").agg(
        [
            pl.len().alias("user.shows"),
            pl.col("viewtime_float").mean().alias("user.avg_timespent"),
            ((pl.col("like").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.like_ctr"
            ),
            ((pl.col("share").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.share_ctr"
            ),
            ((pl.col("bookmark").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.bookmark_ctr"
            ),
            ((pl.col("open_comments").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.comments_ctr"
            ),
            ((pl.col("dislike").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.dislike_ctr"
            ),
            ((pl.col("survival_15").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.heartbeat15_rate"
            ),
            ((pl.col("survival_30").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "user.heartbeat30_rate"
            ),
        ]
    )

    item_stats = df_history.group_by("item_id").agg(
        [
            pl.len().alias("item.shows"),
            pl.col("viewtime_float").mean().alias("item.avg_timespent"),
            ((pl.col("like").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.like_ctr"
            ),
            ((pl.col("share").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.share_ctr"
            ),
            ((pl.col("bookmark").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.bookmark_ctr"
            ),
            ((pl.col("open_comments").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.comments_ctr"
            ),
            ((pl.col("dislike").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.dislike_ctr"
            ),
            ((pl.col("click_on_author").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.click_on_author_ctr"
            ),
            ((pl.col("survival_15").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.heartbeat15_rate"
            ),
            ((pl.col("survival_30").sum() + ALPHA_SMOOTH) / (pl.len() + BETA_SMOOTH)).alias(
                "item.heartbeat30_rate"
            ),
        ]
    )

    users_meta = pl.read_parquet(f"{raw_dir}/metadata/users_metadata.parquet").select(
        ["user_id", "age", "gender", "geo"]
    )
    items_meta = pl.read_parquet(f"{raw_dir}/metadata/items_metadata.parquet").select(
        ["item_id", "duration"]
    )

    ials_viewtime = train_ials(df_history, "viewtime_float")
    ials_feedback = train_ials(df_history, "any_feedback")
    ials_survival = train_ials(df_history, "survival_15")

    def build_dataset(df_target: pl.DataFrame, name: str):
        df = df_target.join(users_meta, on="user_id", how="left", maintain_order="left")
        df = df.join(items_meta, on="item_id", how="left", maintain_order="left")

        df = df.join(user_stats, on="user_id", how="left", maintain_order="left")
        df = df.join(item_stats, on="item_id", how="left", maintain_order="left")

        df = df.with_columns(cs.numeric().fill_null(-1000.0))

        df = apply_ials(df, *ials_viewtime, "viewtime")
        df = apply_ials(df, *ials_feedback, "feedback")
        df = apply_ials(df, *ials_survival, "survival")

        num_cols = [
            "duration",
            "user.shows",
            "user.avg_timespent",
            "user.like_ctr",
            "user.share_ctr",
            "user.bookmark_ctr",
            "user.comments_ctr",
            "user.dislike_ctr",
            "user.heartbeat15_rate",
            "user.heartbeat30_rate",
            "item.shows",
            "item.avg_timespent",
            "item.like_ctr",
            "item.share_ctr",
            "item.bookmark_ctr",
            "item.comments_ctr",
            "item.dislike_ctr",
            "item.click_on_author_ctr",
            "item.heartbeat15_rate",
            "item.heartbeat30_rate",
            "ials.viewtime.dot",
            "ials.viewtime.cos",
            "ials.viewtime.user_norm",
            "ials.viewtime.item_norm",
            "ials.feedback.dot",
            "ials.feedback.cos",
            "ials.feedback.user_norm",
            "ials.feedback.item_norm",
            "ials.survival.dot",
            "ials.survival.cos",
            "ials.survival.user_norm",
            "ials.survival.item_norm",
        ]

        cat_cols = ["age", "gender", "geo", "place", "platform", "agent"]

        final_df = df.select(
            [
                (pl.col("timespent").clip(0, NORM_CONSTANT) / NORM_CONSTANT).alias(
                    "consumption_time"
                ),
                pl.concat_list([pl.col(c).cast(pl.Float32) for c in num_cols]).alias(
                    "num_features"
                ),
                pl.concat_list([pl.col(c).cast(pl.Int64) for c in cat_cols]).alias("cat_features"),
            ]
        )

        final_df.write_parquet(f"{final_out_dir}/{name}.parquet")

    build_dataset(df_train, "train")
    build_dataset(df_val, "val")
    build_dataset(df_test, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess VK-LSVD dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw/VK-LSVD", help="Path to raw data")
    parser.add_argument(
        "--out_dir", type=str, default="data/processed/VK-LSVD", help="Output directory"
    )
    args = parser.parse_args()

    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
