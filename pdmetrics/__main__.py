from rich import print
from pdmetrics.syn.data import pdData
from pdmetrics.syn.classification import Classification
from pdmetrics.metrics.f1 import pdF1

if __name__ == "__main__":
    columns = ["image_id", "image_file", "mask_file"]
    data = pdData.from_columns(columns)
    print(data)

    cc = Classification(shape=[2, 3], num_classes=2)
    example = cc.get_preds_target(mtype="random")
    print(example)

    db_path = "/tmp/f1.db"
    f1 = pdF1(db_path)
    print(f1)

    stats = f1.compute_over_example(example)
    print(stats)
