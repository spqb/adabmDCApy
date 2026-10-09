# Retrain with experimental feedback · `reintegrate`

When some generated or designed sequences have been tested experimentally, their results can be fed back into training, as in [Calvanese et al. (2025)](https://arxiv.org/abs/2504.01593). Sequences that worked pull the model toward them; sequences that failed push it away. In that study, reintegration raised the fraction of functional designs.

```bash
adabmDCA reintegrate -d natural.fasta --reint tested.fasta --adj adjustments.txt -o reintegrated
```

## Inputs

| Option | Meaning |
| --- | --- |
| `-d` | Natural alignment, as for `train` |
| `--reint` | Tested sequences, aligned like the natural ones |
| `--adj` | Text file with one number per tested sequence, in the same order: `1` for a sequence that passed the test, `-1` for one that failed. Values in between grade the outcome |
| `--lambda_` | Strength of the experimental term. Default `1 / max|adj|`, which is 1 for ±1 adjustments |

The PCD training options of [`train`](train.md) are accepted (model type, sweeps, chains, validation set, …). A good starting point is ±1 adjustments with `--lambda_ 1`; raise \(\lambda\) to give the experiments more influence over the natural data.

!!! note "Reintegration always trains with PCD"
    The experimental term gives negative weights to failed sequences, which PTT does not support. `reintegrate` therefore always trains with PCD and does not offer `--strategy` or the `--ptt-*` options. Check the result by sampling it with `sample --strategy pcd` and making sure it mixes.

## What it does

The natural sequences keep their usual weights \(w_m\), normalized by \(M_{\rm eff}\). Each tested sequence \(r\) gets the weight \(\lambda\,a_r/N_{\rm exp}\) relative to the natural data, where \(a_r\) is its adjustment and \(N_{\rm exp}\) the number of tested sequences. The model is then trained on the combined, signed weights, which amounts to maximizing the likelihood of the natural sequences plus \(\lambda\) times the average of \(a_r \log p(y_r)\) over the tested ones. See [the derivation](../algorithms/analysis.md#experimental-reintegration).

With strong negative adjustments, some combined pair frequencies can become negative; they are clipped to zero before the pseudocount is added. The model then departs from the formal objective, so very large \(\lambda\) values are not recommended.

## Outputs

The usual training outputs ([`train`](train.md#outputs)), plus the combined alignment and its signed weights, named after the resolved \(\lambda\) (for example `reintegrated-lambda_1.00_msa.fasta` and `reintegrated-lambda_1.00_weights.dat`), and a reintegration summary in JSON. Check that the adjustments line up with the tested sequences in the combined files before interpreting the model.

## In Python

```python
import numpy as np
from adabmDCA import load_alignment, reintegrate_model

result = reintegrate_model(
    load_alignment("natural.fasta"), load_alignment("tested.fasta"),
    np.loadtxt("adjustments.txt"), output_dir="reintegrated",
)
print(result.lambda_value, result.training.stop_reason)
```
