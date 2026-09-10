from pathlib import Path

import torch

from adabmDCA.io import load_params, save_params

TOKENS = "AB-"


def _symmetric_params(length: int = 3):
    generator = torch.Generator().manual_seed(12)
    bias = torch.randn((length, len(TOKENS)), generator=generator)
    raw = torch.randn((length, len(TOKENS), length, len(TOKENS)), generator=generator)
    couplings = 0.5 * (raw + raw.permute(2, 3, 0, 1))
    positions = torch.arange(length)
    couplings[positions, :, positions, :] = 0
    return {"bias": bias, "coupling_matrix": couplings}


def test_text_roundtrip_with_symmetric_mask_writes_one_triangle(tmp_path: Path):
    params = _symmetric_params()
    mask = torch.ones_like(params["coupling_matrix"], dtype=torch.bool)
    positions = torch.arange(len(params["bias"]))
    mask[positions, :, positions, :] = False
    path = tmp_path / "params.dat"

    save_params(str(path), params, tokens=TOKENS, mask=mask)
    loaded = load_params(str(path), tokens=TOKENS, device=torch.device("cpu"))

    records = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    coupling_records = [parts for parts in records if parts[0] == "J"]
    assert coupling_records
    assert all(int(parts[1]) < int(parts[2]) for parts in coupling_records)
    torch.testing.assert_close(loaded["bias"], params["bias"], rtol=0, atol=0)
    torch.testing.assert_close(loaded["coupling_matrix"], params["coupling_matrix"], rtol=0, atol=0)


def test_loader_preserves_legacy_files_with_both_coupling_triangles(tmp_path: Path):
    path = tmp_path / "legacy_params.dat"
    path.write_text(
        "J 0 1 A B 1.25\n"
        "J 1 0 B A 2.75\n"
        "h 0 A 0.5\n"
        "h 0 B 0.0\n"
        "h 0 - 0.0\n"
        "h 1 A 0.0\n"
        "h 1 B -0.5\n"
        "h 1 - 0.0\n",
        encoding="utf-8",
    )

    loaded = load_params(str(path), tokens=TOKENS, device=torch.device("cpu"))

    assert loaded["coupling_matrix"][0, 0, 1, 1].item() == 4.0
    assert loaded["coupling_matrix"][1, 1, 0, 0].item() == 4.0


def test_parameter_io_honors_requested_dtype(tmp_path: Path):
    path = tmp_path / "params.dat"
    params = _symmetric_params(length=2)
    save_params(str(path), params, tokens=TOKENS)

    loaded = load_params(
        str(path),
        tokens=TOKENS,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert loaded["bias"].dtype == torch.float64
    assert loaded["coupling_matrix"].dtype == torch.float64
