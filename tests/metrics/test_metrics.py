from rich import print
from pdmetrics.metrics.base import pdMetrics


def test_create():
    print()
    metrics = pdMetrics(
        name="f1",
        tracked=["f1", "tp", "fp", "fn"],
        db_path="/tmp/pdmetrics.db",
    )
    print(metrics)
