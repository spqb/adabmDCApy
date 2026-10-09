# Plot a run · `plot-training-log`

Draw the curves recorded by a training run, finished or still running:

```bash
adabmDCA plot-training-log model/history.csv
```

The argument can be `history.csv`, `<label>_history.csv`, or the run's log file, which points to its history. `train --plot-training-logs` does the same automatically when training ends.

## Outputs

One PNG per recorded quantity, named `<label>_<quantity>.png` (`training_pearson.png` without a label), and an overview combining all of them (`<label>_overview.png`), in `<label>_plots/` next to the history table (`training_plots/` without a label; choose another folder with `-o`).

| Quantity | Plots | Available for |
| --- | --- | --- |
| `pearson` | Pearson and slope of the connected correlations, training and validation, with the target | all runs |
| `loglikelihood` | Log-likelihood per site, training and validation | PTT |
| `entropy` | Model entropy | PTT |
| `density` | Fraction of active couplings | sparse models |
| `ladder` | Active replicas and models saved along the trajectory | PTT |
| `acceptance` | Smallest swap acceptance between neighbouring models | PTT |
| `learning_rates` | Field and coupling learning rates | PTT adaptive |
| `kl` | Predicted KL divergence of each update | PTT adaptive |
| `lag` | Lag of the chains behind the model | PTT adaptive |

When `events.jsonl` and `training.json` are present, the validation stop, lag pauses and recoveries are marked on the curves. History tables from older versions of the package are read too.

[Training: reading the plots](../algorithms/training.md#reading-the-training-plots) shows each figure on a real run and says what healthy and unhealthy curves look like.

## In Python

```python
import pandas as pd

history = pd.read_csv("model/history.csv")
ax = history.plot(x="step", y=["pearson", "pearson_val"])
ax.figure.savefig("pearson.png")
```

A `TrainingResult` returned by `train_model` gives the same table with `result.history_dataframe()`.
