#src/rsna_build_meta.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SUBTYPES = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def main():
    data_dir = Path("data/rsna")
    train_csv = data_dir / "stage_2_train.csv"
    dicom_dir = data_dir / "stage_2_train"

    assert train_csv.exists(), f"{train_csv} not found"
    assert dicom_dir.exists(), f"{dicom_dir} not found"

    print("Loading stage_2_train.csv ...")
    df = pd.read_csv(train_csv)

    print("Parsing IDs to image_id and subtype ...")
    df["image_id"] = df["ID"].str.rsplit("_", n=1).str[0]
    df["subtype"] = df["ID"].str.rsplit("_", n=1).str[1]

    print("Pivoting to multi-label format ...")
    pivot = (
        df.pivot_table(
            index="image_id",
            columns="subtype",
            values="Label",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    for st in SUBTYPES:
        if st not in pivot.columns:
            pivot[st] = 0

    pivot["filepath"] = pivot["image_id"].apply(
        lambda x: str(dicom_dir / f"{x}.dcm")
    )

    print("Checking which DICOM files exist ...")
    pivot["exists"] = pivot["filepath"].apply(lambda p: Path(p).exists())
    missing = (pivot["exists"] == False).sum()
    if missing > 0:
        print(
            f"Warning: {missing} entries have missing DICOM files, dropping them.")
    pivot = pivot[pivot["exists"]].drop(columns=["exists"])

    cols = ["image_id", "filepath"] + SUBTYPES
    pivot = pivot[cols]

    meta_csv = data_dir / "rsna_meta.csv"
    pivot.to_csv(meta_csv, index=False)
    print(f"Saved full metadata to {meta_csv}, rows = {len(pivot)}")

    train_df, val_df = train_test_split(
        pivot,
        test_size=0.1,
        random_state=42,
        stratify=pivot["any"], 
    )

    train_csv_out = data_dir / "rsna_train.csv"
    val_csv_out = data_dir / "rsna_val.csv"
    train_df.to_csv(train_csv_out, index=False)
    val_df.to_csv(val_csv_out, index=False)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    print(f"Saved {train_csv_out} and {val_csv_out}")


if __name__ == "__main__":
    main()
