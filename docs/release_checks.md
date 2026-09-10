# Release verification

Before publishing a release, require successful **Test adabmDCA** and **GPU
release gate** workflow runs for the exact release commit. Both workflows run
on pushes to the maintained release branches, on `v*` tags, and manually.
CPU tests also run for pull requests targeting those branches.

The CPU workflow installs the declared development dependencies and uses
pytest to discover both unittest classes and pytest functions. Its artifacts
include the test report and CLI smoke-test outputs.

## GPU runner setup

Register a Linux x64 GitHub Actions runner with the labels `self-hosted`,
`linux`, `x64`, and `gpu`. It needs an NVIDIA Ampere-or-newer GPU, a compatible
NVIDIA driver, and network access to install Python dependencies. The workflow
creates its own Python environment. If the repository uses different runner
labels, update `.github/workflows/gpu.yml` accordingly.

The GPU workflow deliberately runs only on branch/tag pushes and manual
dispatches. Keep those refs and dispatch access restricted to trusted
maintainers; do not run unreviewed fork code on a persistent self-hosted runner.

`pytest --require-gpu` fails immediately without usable CUDA, Triton, and
bfloat16 support. It also fails if any test is skipped. The full suite includes
controlled sampler comparisons, seeded trajectories and RNG state, strided
inputs, parameter-cache invalidation, and mixed-precision training tests.
The `gpu-release-evidence` artifact records the commit, test results, dependency
versions, and hardware/runtime details.

Without a matching runner, the GPU job remains queued; it is not a passing
release check. Provisioning the runner and configuring repository rules are
GitHub administration steps, not changes this workflow can perform. This
repository has no automated package publisher: maintainers must verify both
checks before publishing, and any future publishing job must depend on them.

For local verification on a suitable GPU:

```bash
uv sync --locked
uv run python -m pytest -q --require-gpu
```
