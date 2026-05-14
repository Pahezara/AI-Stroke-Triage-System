#src/isles_build_meta.py

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    root = Path("data/isles")
    raw_root = root / "rawdata"
    der_root = root / "derivatives"

    assert raw_root.exists(), f"{raw_root} missing"
    assert der_root.exists(), f"{der_root} missing"

    cases = []

    for sub_dir in sorted(raw_root.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            case_id = f"{sub_dir.name}_{ses_dir.name}"

            anat_dir = ses_dir / "anat"
            dwi_dir = ses_dir / "dwi"

            flair = next(anat_dir.glob("*_FLAIR.nii.gz"), None)
            dwi = next(dwi_dir.glob("*_dwi.nii.gz"), None)
            adc = next(dwi_dir.glob("*_adc.nii.gz"), None)

            der_ses_dir = der_root / sub_dir.name / ses_dir.name
            msk = next(der_ses_dir.glob("*_msk.nii.gz"), None)

            if not flair or not dwi or not adc or not msk:
                print(
                    f"[WARN] Skipping {case_id}: missing one of FLAIR/DWI/ADC/MSK")
                continue

            cases.append({
                "case_id": case_id,
                "flair": str(flair),
                "dwi": str(dwi),
                "adc": str(adc),
                "mask": str(msk),
            })

    if len(cases) == 0:
        raise RuntimeError(
            "No valid ISLES cases found. Check folder structure.")

    df = pd.DataFrame(cases)
    df.to_csv(root / "isles_meta.csv", index=False)
    print(f"Saved metadata for {len(df)} cases → data/isles/isles_meta.csv")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True)

    def make_list(rows):
        lst = []
        for _, r in rows.iterrows():
            lst.append({
                "case_id": r["case_id"],
                "dwi": r["dwi"],
                "adc": r["adc"],
                "flair": r["flair"],
                "label": r["mask"],
            })
        return lst

    train_list = make_list(train_df)
    val_list = make_list(val_df)

    (root / "isles_train_list.json").write_text(json.dumps(train_list, indent=2))
    (root / "isles_val_list.json").write_text(json.dumps(val_list, indent=2))

    print(f"Train cases: {len(train_list)}")
    print(f"Val cases:   {len(val_list)}")
    print("Lists saved → data/isles/isles_train_list.json, isles_val_list.json")


if __name__ == "__main__":
    main()
