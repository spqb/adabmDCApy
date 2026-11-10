"""PTT on CPU must work when the optional Numba dependency is missing."""

import subprocess
import sys
import textwrap


def test_ptt_sampler_and_archive_without_numba(tmp_path):
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules["numba"] = None  # makes ``import numba`` raise ImportError
        import torch
        from adabmDCA.ptt.sampler import PTTSampler
        from adabmDCA.numba_kernels import is_numba_available

        assert not is_numba_available()
        params = {{"bias": torch.zeros(5, 4), "coupling_matrix": torch.zeros(5, 4, 5, 4)}}
        sampler = PTTSampler(params, tokens="ACDE", n_chains=8, sampler="metropolized_gibbs", seed=0)
        path = r"{tmp_path / 'ptt.h5'}"
        sampler.save_archive(path)
        PTTSampler.from_archive(path, device="cpu", mode="inspect")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
