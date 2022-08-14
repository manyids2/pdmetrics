# pdmetrics

Metrics for various ML tasks with pandas DataFrames.

## Installation

```bash
# # From pip directly
# pip install pdmetrics

# Dev version
pip install -e .
```

## Usage

Typical metrics usage when one summary score is required.

```python

# Get some preds and target data
from pdmetrics.syn import Classification

cc = Classification(shape=[2, 3], num_classes=2)
example = cc.get_random()
cc.print(example)

# Compute the metrics
from pdmetrics.metrics import pdF1

metrics = pdF1('/tmp/f1.db')

stats = metrics.one_example(example)
metrics.print(stats)

# Check the dataframe
metrics.print()

```

Typical metrics usage for exploration.

```python

# Get some preds and target data
from pdmetrics.syn import Classification

cc = Classification(shape=[2, 3], num_classes=2)
example = cc.get_random()
cc.print(example)

# Compute the metrics
from pdmetrics.metrics import pdF1

metrics = pdF1('/tmp/f1.db')

stats = metrics.one_example(example)
metrics.print(stats)

# Check the dataframe
metrics.print()

```

## Exploration of saved metrics
