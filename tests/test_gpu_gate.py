"""Exercise GPU gate failures on CPU as well as on GPU CI."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from conftest import _NoSkippedTests
from conftest import pytest_configure as configure_gpu_gate


@pytest.mark.parametrize("cuda,triton,capability,bfloat16", [
    (False, True, 8, True), (True, False, 8, True),
    (True, True, 7, True), (True, True, 8, False),
])
def test_gate_rejects_missing_hardware(cuda, triton, capability, bfloat16):
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("adabmDCA.sampling_triton.is_triton_available", return_value=triton),
        patch("torch.cuda.get_device_capability", return_value=(capability, 0)),
        patch("torch.cuda.is_bf16_supported", return_value=bfloat16),
        pytest.raises(pytest.UsageError, match="requires"),
    ):
        configure_gpu_gate(Mock())


def test_gate_registers_skip_enforcement_on_supported_hardware():
    config = Mock()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("adabmDCA.sampling_triton.is_triton_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        configure_gpu_gate(config)
    assert isinstance(config.pluginmanager.register.call_args.args[0], _NoSkippedTests)


@pytest.mark.parametrize("hook", ["pytest_collectreport", "pytest_runtest_logreport"])
def test_gate_rejects_skips_even_without_terminal_reporter(hook):
    gate = _NoSkippedTests()
    session = SimpleNamespace(config=Mock(), exitstatus=pytest.ExitCode.OK)
    session.config.pluginmanager.get_plugin.return_value = None
    getattr(gate, hook)(SimpleNamespace(skipped=True))
    gate.pytest_runtest_logreport(SimpleNamespace(skipped=False))
    gate.pytest_sessionfinish(session, pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_gate_preserves_existing_exit_status_without_skips():
    gate = _NoSkippedTests()
    session = SimpleNamespace(exitstatus=pytest.ExitCode.INTERRUPTED)
    gate.pytest_sessionfinish(session, session.exitstatus)
    assert session.exitstatus == pytest.ExitCode.INTERRUPTED
