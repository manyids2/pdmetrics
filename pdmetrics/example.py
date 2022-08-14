import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rich import print as _print
from pdmetrics.syn.data import save_df
from pdmetrics.syn.classification import Classification
from pdmetrics.metrics.f1 import pdF1
from pdmetrics.utils.draw import mask_to_colors, COLORS, to_pil_image

if __name__ == "__main__":
    cc = Classification(shape=[128, 128], num_classes=2)

    labels = {0: "false", 1: "true"}
    db_path = "/tmp/classification/metrics.db"
    f1 = pdF1(db_path, labels)
    _print(f1)

    # Create a dataset
    data_dir = Path("/tmp/classification")
    data_dir.mkdir(exist_ok=True)
    [(data_dir / d).mkdir(exist_ok=True) for d in ["preds", "target"]]

    num_examples = 128
    examples = {}
    rows = {}
    pbar = tqdm(range(num_examples), total=num_examples, ncols=60)
    for idx in pbar:
        example = cc.get_preds_target("random")

        # Serialize actual data
        preds = example["preds"]
        target = example["target"]
        preds_file = data_dir / "preds" / f"{idx}.npz"
        target_file = data_dir / "target" / f"{idx}.npz"
        np.savez(preds_file, mask=preds, format="binary-mask")
        np.savez(target_file, mask=target, format="binary-mask")

        # Serialize visualization
        preds_overlay = mask_to_colors(preds, COLORS)
        target_overlay = mask_to_colors(target, COLORS)
        preds_overlay_file = data_dir / "preds" / f"{idx}.png"
        target_overlay_file = data_dir / "target" / f"{idx}.png"
        to_pil_image(preds_overlay).save(str(preds_overlay_file))
        to_pil_image(target_overlay).save(str(target_overlay_file))

        examples[idx] = example
        rows[idx] = {
            "rowid": idx,
            "preds_file": str(preds_file),
            "target_file": str(target_file),
            "preds_overlay_file": str(preds_overlay_file),
            "target_overlay_file": str(target_overlay_file),
        }

    df = pd.DataFrame(list(rows.values()))
    save_df(df, data_dir / "data.db", "dataset", verbose=True)

    _labels = pd.DataFrame([{"class_idx": k, "label": v} for k, v in labels.items()])
    save_df(_labels, data_dir / "data.db", "labels", verbose=True)

    pbar = tqdm(df.iterrows(), total=len(df), ncols=60)
    for _, row in pbar:
        example = examples[row.rowid]
        f1.compute_over_example(example, row.rowid)
    save_df(df, f1.db_path, "dataset", verbose=True)
    save_df(f1.df, f1.db_path, "f1", index=True, verbose=True)

    save_df(_labels, f1.db_path, "labels", verbose=True)

    _threshold = pd.DataFrame([{"threshold": f1.threshold}])
    save_df(_threshold, f1.db_path, "threshold", verbose=True)
