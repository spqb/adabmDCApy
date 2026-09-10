"""Explicit hardware requirements for the GPU release gate."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu", action="store_true", default=False,
        help="Require Ampere+ CUDA and Triton, and fail if any tests are skipped.",
    )


def pytest_configure(config):
    if not config.getoption("--require-gpu"):
        return
    import torch

    from adabmDCA.sampling_triton import is_triton_available

    if not torch.cuda.is_available() or not is_triton_available():
        raise pytest.UsageError("--require-gpu requires usable CUDA and Triton.")
    if torch.cuda.get_device_capability()[0] < 8 or not torch.cuda.is_bf16_supported():
        raise pytest.UsageError("--require-gpu requires an Ampere-or-newer GPU with bfloat16 support.")
    config.pluginmanager.register(_NoSkippedTests(), "gpu-no-skips")


class _NoSkippedTests:
    """Track reports directly so the gate also works without a terminal reporter."""

    skipped = False

    def pytest_collectreport(self, report):
        self.skipped |= report.skipped

    def pytest_runtest_logreport(self, report):
        self.skipped |= report.skipped

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if not self.skipped:
            return
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "GPU release gate failed: tests were skipped")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
