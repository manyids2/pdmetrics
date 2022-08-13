from rich import print
from pdmetrics.syn.classification import Classification


def test_create():
    print()
    shape = [2, 5]
    num_classes = 5
    cc = Classification(shape, num_classes)
    print(cc)
    assert isinstance(cc, Classification)


def test_all_zeros():
    print()
    shape = [2, 5]
    num_classes = 5
    cc = Classification(shape, num_classes)
    batch = cc.get_preds_target("all_zeros")
    assert batch["preds"].sum() == 0
    assert batch["target"].sum() == 0


def test_all_correct():
    print()
    shape = [2, 5]
    num_classes = 5
    cc = Classification(shape, num_classes)
    batch = cc.get_preds_target("all_correct")
    assert (batch["preds"] != batch["target"]).sum() == 0


def test_all_wrong():
    print()
    shape = [2, 5]
    num_classes = 5
    cc = Classification(shape, num_classes)
    batch = cc.get_preds_target("all_wrong")
    assert (batch["preds"] == batch["target"]).sum() == 0
