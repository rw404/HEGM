import os
import argparse
import polars as pl
import polars.selectors as cs
import json


def fix_corrupted_csv(input_path, output_path):
    if not os.path.exists(output_path):
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def get_norm_constant(path_items):
    df_items = pl.read_csv(path_items, infer_schema_length=10000)
    df_items = df_items.unique(subset=["video_id"])
    df_items = df_items.filter(pl.col("video_duration") > 0)

    if "video_type" in df_items.columns:
        df_items = df_items.filter(pl.col("video_type") != "AD")

    p99 = df_items.select(pl.col("video_duration").quantile(0.99)).item()
    return p99


def main(raw_dir: str, out_dir: str):
    fix_corrupted_csv(
        f"{raw_dir}/kuairec_caption_category.csv",
        f"{raw_dir}/cleaned_kuairec_caption_category.csv",
    )
    NORM_CONSTANT = get_norm_constant(f"{raw_dir}/item_daily_features.csv")

    df1 = pl.read_csv(f"{raw_dir}/big_matrix.csv", infer_schema_length=10000)
    df2 = pl.read_csv(f"{raw_dir}/user_features.csv", infer_schema_length=10000)
    df3 = (
        pl.read_csv(f"{raw_dir}/item_daily_features.csv", infer_schema_length=10000)
        .select(["video_id", "video_type", "music_id", "video_tag_id"])
        .unique(subset=["video_id"])
    )
    df4 = pl.read_csv(f"{raw_dir}/item_categories.csv", infer_schema_length=10000)
    df5 = pl.read_csv(
        f"{raw_dir}/cleaned_kuairec_caption_category.csv", infer_schema_length=10000
    ).select(
        [
            "video_id",
            "first_level_category_id",
            "second_level_category_id",
            "third_level_category_id",
        ]
    )

    df1 = df1.with_columns(
        pl.col("user_id").cast(pl.Int64, strict=False),
        pl.col("video_id").cast(pl.Int64, strict=False),
    )
    df2 = df2.with_columns(pl.col("user_id").cast(pl.Int64, strict=False))
    df3 = df3.with_columns(pl.col("video_id").cast(pl.Int64, strict=False))
    df4 = df4.with_columns(pl.col("video_id").cast(pl.Int64, strict=False))
    df5 = df5.with_columns(pl.col("video_id").cast(pl.Int64, strict=False))

    df = df1.join(df2, on="user_id", how="left")
    df = df.join(df3, on="video_id", how="left")
    df = df.join(df4, on="video_id", how="left")
    df = df.join(df5, on="video_id", how="left")

    df = df.with_columns([cs.string().fill_null("UNK"), cs.numeric().fill_null(0)])

    df = df.filter(
        (pl.col("video_type") != "AD")
        & (pl.col("play_duration") > 0)
        & (pl.col("video_duration") > 0)
    )

    df = df.with_columns(
        [
            (pl.col("video_duration").clip(0, NORM_CONSTANT) / NORM_CONSTANT).alias(
                "video_duration_norm"
            ),
            (pl.col("play_duration").clip(0, NORM_CONSTANT) / NORM_CONSTANT).alias(
                "consumption_time"
            ),
        ]
    )

    df = df.sort(["timestamp", "user_id", "video_id"])

    train_size = int(0.8 * df.height)
    val_size = int(0.1 * df.height)

    train_df = df.slice(0, train_size)
    val_df = df.slice(train_size, val_size)
    test_df = df.slice(train_size + val_size, df.height)

    user_mapping = (
        train_df.select("user_id").unique().sort("user_id").with_row_index("user_id_encoded")
    )
    item_mapping = (
        train_df.select("video_id").unique().sort("video_id").with_row_index("item_id_encoded")
    )

    num_users = user_mapping.height
    num_items = item_mapping.height

    def encode_ids(df_split, is_train=False):
        join_how = "inner" if not is_train else "left"
        res = df_split.join(user_mapping, on="user_id", how=join_how)
        res = res.join(item_mapping, on="video_id", how=join_how)
        return res

    train_df = encode_ids(train_df, is_train=True)
    val_df = encode_ids(val_df, is_train=False)
    test_df = encode_ids(test_df, is_train=False)

    num_cols = ["video_duration_norm", "timestamp"]
    cat_cols = [
        "user_active_degree",
        "is_lowactive_period",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "first_level_category_id",
        "second_level_category_id",
        "third_level_category_id",
    ] + [f"onehot_feat{i}" for i in range(18)]

    for col in cat_cols:
        mapping = train_df.select(col).unique().sort(col).with_row_index(f"{col}_encoded")
        unk_token = mapping.height

        train_df = train_df.join(mapping, on=col, how="left").with_columns(
            pl.col(f"{col}_encoded").fill_null(unk_token)
        )
        val_df = val_df.join(mapping, on=col, how="left").with_columns(
            pl.col(f"{col}_encoded").fill_null(unk_token)
        )
        test_df = test_df.join(mapping, on=col, how="left").with_columns(
            pl.col(f"{col}_encoded").fill_null(unk_token)
        )

    def format_to_final(df_split):
        encoded_cat_cols = [f"{c}_encoded" for c in cat_cols]
        return df_split.select(
            [
                pl.col("consumption_time").cast(pl.Float32),
                pl.col("user_id_encoded").cast(pl.Int64).alias("user_id"),
                pl.col("item_id_encoded").cast(pl.Int64).alias("item_id"),
                pl.concat_list([pl.col(c).cast(pl.Float32) for c in num_cols]).alias(
                    "num_features"
                ),
                pl.concat_list([pl.col(c).cast(pl.Int64) for c in encoded_cat_cols]).alias(
                    "cat_features"
                ),
            ]
        )

    final_train = format_to_final(train_df)
    final_val = format_to_final(val_df)
    final_test = format_to_final(test_df)

    os.makedirs(out_dir, exist_ok=True)

    final_train.write_parquet(f"{out_dir}/train.parquet")
    final_val.write_parquet(f"{out_dir}/val.parquet")
    final_test.write_parquet(f"{out_dir}/test.parquet")

    meta_info = {
        "NORM_CONSTANT": float(NORM_CONSTANT),
        "num_users": num_users,
        "num_items": num_items,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KuaiRec dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw/KuaiRec", help="Path to raw data")
    parser.add_argument(
        "--out_dir", type=str, default="data/processed/KuaiRec", help="Output directory"
    )
    args = parser.parse_args()

    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
