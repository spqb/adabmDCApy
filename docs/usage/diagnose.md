# Diagnose a run · `diagnose`

Read the files of a training run and say whether it is healthy, and what to change if not:

```bash
adabmDCA diagnose model                                  # a finished or running run
adabmDCA diagnose model -s check/sampling.json           # with the check of the finished model
adabmDCA diagnose model -l run1 --json                   # a labelled run, full report as JSON
```

It reads `history.csv`, `events.jsonl`, `training.json` and the log of the run in the folder (with `-l`, the files prefixed by that label), and, with `-s`, the `sampling.json` written by `adabmDCA sample --plot` on the finished model. It prints a verdict and the findings behind it:

| Verdict | Meaning |
| --- | --- |
| `healthy` | Nothing above an informational note |
| `needs_attention` | At least one warning, such as a run that hit its budget or a sampled Pearson well below the training one |
| `failed` | An error: the run stopped or stalled on trapped configurations |
| `running` | No summary yet and the files changed in the last 15 minutes |
| `incomplete` | No summary and no recent output: the run was cancelled or crashed |

Each finding names the failure of the [diagnostics guide](../algorithms/diagnostics.md) it points to (F1–F8) and the change that guide recommends, as command-line options. The thresholds are the guide's heuristics, so the report also lists the indicators they were computed from.

From Python, `adabmDCA.diagnose_run(run_dir, label=None, sampling_json=None)` returns the same report as a dictionary.
