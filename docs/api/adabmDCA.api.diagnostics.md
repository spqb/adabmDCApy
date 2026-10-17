<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/diagnostics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.diagnostics`
Turn the files of a training run (and optionally a sampling check) into a verdict.

The rules follow the "Diagnose a run" guide (docs/algorithms/diagnostics.md): each finding names the failure it points to (F1-F8) and the change the guide recommends, written as CLI options. Thresholds are the guide's; they are heuristics, so findings say what was measured.

**Global Variables**
---------------
- **GUIDE**
- **FAILURES**
- **SEVERITY_ORDER**
- **STALE_SECONDS**
- **MIN_UPDATES_FOR_RATES**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/diagnostics.py#L471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `diagnose_run`

```python
diagnose_run(
    run_dir: 'str | Path',
    label: 'str | None' = None,
    sampling_json: 'str | Path | None' = None
) → dict[str, Any]
```

Indicators and findings for a training run directory, and optionally a sampling check.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
