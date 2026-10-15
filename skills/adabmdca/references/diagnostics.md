# Is the run healthy? What to change if not

Training metrics come from the same chains that estimate the gradient. If those chains fall out of equilibrium, the model is wrong **and** the metrics can still look fine. So every model that will be relied on must be checked by independent sampling. Full guide: https://spqb.github.io/adabmDCApy/algorithms/diagnostics/ (`docs/algorithms/diagnostics.md` in a clone).

## 1. How did training end?

Read `training.json` → `data.stop_reason`, `data.converged`, `data.warnings`, or the `[END]` section of `adabmDCA.log`.

- `validation_plateau` or `target_pearson`: finished; go to step 3.
- A budget (`max_gradient_steps`, …): usable but not converged. Say so; resume with `--ptt-resume model/ptt.h5 --max-gradient-steps N` if the user wants more.
- An error: the message names the cause; see the decision table.
- No `training.json`, no new rows in `history.csv` for a long time: still running or stuck; step 2.

## 2. While it runs, or when it seems stuck

```bash
tail -n 3 model/history.csv
grep '»' model/adabmDCA.log | tail -n 20
```

The terminal status names the PTT phase: `optimizing` is normal; `measuring mixing`, `mixing warmup`, `reservoir warmup`, `reservoir renewal`, `collecting reservoir`, `equilibrating replicas`, `recovery warmup` are checks. A phase stuck for tens of minutes at the same update, or wall time growing far faster than `time_s`, is the signature of trapped configurations (F1). Long PTT runs on clustered families are normal: slow is not stuck.

Healthy `history.csv`:

| Column | Healthy | Warning |
| --- | --- | --- |
| `pearson`, `pearson_val` | Rising, then flat | Sudden drops after a recovery |
| `ll_val` | Rising, then flat | Falling over hundreds of updates |
| `acceptance_min` | Saw-tooth ~0.25 to 0.5–0.9 | Long stretches below 0.25 or 0.1 |
| `replicas` | 2–3 | |
| `lr_bias`, `lr_coupling` | At `--lr` most of the time | Halved after recoveries; `lr_bias` at the floor |
| `kl` | Median ~1e-3 | Pinned at 0.01 |
| `lag_drift_tail` | Below ~0.05 | Peaks near 0.25 |

## 3. Check the finished model

```bash
adabmDCA sample -p model/ptt.h5 -d train.fasta -v validation.fasta --plot -o check
```

1. `sampling.json` → `ptt_diagnostics.converged` must be true. If not, do not use the samples.
2. Sampled Pearson (`ptt_diagnostics.final_pearson`) vs last training `pearson`, at the same number of sequences as training chains: a healthy model loses 0.003–0.007; more than ~0.01 deserves attention.
3. `ptt_ladder_health.png`: trapped-configuration bars above 2 flag stuck links.
4. `pca_1_2.png`, `data_vs_samples.png`: every data cluster covered, share ratios roughly ×0.8–×1.25.
5. `distances.png` (needs `-v`): generated sequences not closer to training than held-out ones; none identical to a training sequence.

PCD models (no ladder): `sample --strategy pcd -p model_pcd/params.dat.gz -d train.fasta --plot --max_nsweeps 10000`. The run must report a mixing time (not hit `--max_nsweeps`), and `pearson_sampling.png` must rise and settle; a rise then decline means trained out of equilibrium. If either fails, retrain with PTT.

## Decision table

| You see | Do |
| --- | --- |
| Error recommending more local sweeps, or log lines "configurations cannot swap toward the bottom" | Restart from scratch with `--nsweeps 100`. Do not resume |
| Stuck for tens of minutes in a reservoir or mixing phase | Same as above (or wait for the failure message) |
| `» pause: drift tail lag …`, training continues | Nothing; check the finished model |
| Error naming a link "below the target", no immobile configurations | Smaller `--lr` or larger `--ptt-target-acceptance` |
| Error saying "every link accepts at least the target" | More `--nsweeps` |
| Repeated `restored step …; learning rates …` | Read the mixing-failure lines before each recovery |
| `lr_bias` ~1e-14 and `kl` at 0.01 for most updates | Back to `--lr 0.01` |
| Finished, but samples disagree with training (Pearson, clusters) | Retrain with more `--nsweeps` |
| PTT sampling hits `--ptt-max-rounds` or warns that renewal stalled | Usually needs a better training run |
| "PTT initial mixing experiment exceeded its round budget" | Check the input alignment |
| PCD model: MCMC hits `--max_nsweeps` | Retrain with PTT |

Things that do **not** help with trapped configurations: more chains (`--nchains`), and resuming the failed archive with more sweeps.

## Triage script

```python
import json
from collections import Counter
from pathlib import Path
import pandas as pd

def triage(run_dir, label=""):
    p = f"{label}_" if label else ""
    run = Path(run_dir)
    h = pd.read_csv(run / f"{p}history.csv")
    events = [json.loads(l) for l in open(run / f"{p}events.jsonl")]
    counts = Counter(e["event"] for e in events)
    failed = [e for e in events if e["event"] == "ptt_mixing" and e.get("status") not in (None, "converged")]
    summary_file = run / f"{p}training.json"
    s = json.loads(summary_file.read_text())["data"] if summary_file.exists() else {}
    last = h.iloc[-1]
    return {
        "stop_reason": s.get("stop_reason"), "converged": s.get("converged"),
        "updates": int(last.step), "pearson": round(float(last.pearson), 4),
        "best_val_ll": round(float(h.ll_val.max()), 4) if "ll_val" in h else None,
        "failed_mixing_checks": len(failed), "recoveries": counts["ptt_recovery"],
        "lag_pauses": counts["ptt_lag_pause"], "reservoir_refreshes": counts["ptt_reservoir_refreshed"],
        "min_acceptance": round(float(h.acceptance_min.min()), 3) if "acceptance_min" in h else None,
        "median_kl": float(h.kl.median()) if "kl" in h else None,
    }

print(triage("model"))
```

A healthy PTT run: a proper stop reason, no failed checks, no recoveries, minimum acceptance around 0.25, median KL ~1e-3.
