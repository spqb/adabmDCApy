from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from adabmDCA import io
from adabmDCA.exceptions import OutputSerializationError
from adabmDCA.io import load_params, save_params
from adabmDCA.statmech import compute_energy, enumerate_states

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


def test_compressed_parameters_roundtrip_and_are_detected_by_content(tmp_path: Path):
    import gzip

    params = _symmetric_params()
    plain, compressed = tmp_path / "params.dat", tmp_path / "params.dat.gz"
    save_params(str(plain), params, tokens=TOKENS)
    save_params(str(compressed), params, tokens=TOKENS)
    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
        assert handle.read() == plain.read_text(encoding="utf-8")
    expected = load_params(str(plain), tokens=TOKENS, device=torch.device("cpu"))
    # Compression is recognized from the content, so a renamed file loads too.
    renamed = tmp_path / "renamed.dat"
    renamed.write_bytes(compressed.read_bytes())
    for path in (compressed, renamed):
        loaded = load_params(str(path), tokens=TOKENS, device=torch.device("cpu"))
        for key in expected:
            torch.testing.assert_close(loaded[key], expected[key], rtol=0, atol=0)


def test_models_and_alphabet_detection_read_compressed_parameters(tmp_path: Path):
    from argparse import Namespace

    from adabmDCA.api import load_model
    from adabmDCA.scripts._frontend import resolve_alphabet

    path = tmp_path / "params.dat.gz"
    save_params(str(path), _symmetric_params(), tokens=TOKENS)
    model = load_model(path, alphabet=TOKENS, device="cpu")
    assert model.params["bias"].shape == (3, 3)
    rna = tmp_path / "rna.dat.gz"
    save_params(str(rna), {"bias": torch.zeros(2, 5), "coupling_matrix": torch.zeros(2, 5, 2, 5)}, tokens="-ACGU")
    args = Namespace(alphabet="auto", path_params=rna)
    resolve_alphabet(args)
    assert args.alphabet == "rna"


@pytest.mark.parametrize("suffix", [".dat", ".dat.gz"])
def test_open_params_text_modes_are_safe(tmp_path, suffix):
    path = tmp_path / ("params" + suffix)
    with io.open_params(path, "wt") as handle:
        handle.write("h 0 A 1\n")
    original = path.read_bytes()
    for mode in ("r", "rt"):
        with io.open_params(path, mode) as handle:
            assert handle.read() == "h 0 A 1\n"
    for mode in ("rb", "a", "r+", "x", "invalid"):
        with pytest.raises(ValueError, match="mode"):
            io.open_params(path, mode)
    assert path.read_bytes() == original


@pytest.mark.parametrize("suffix", [".dat", ".dat.gz"])
def test_loader_reads_one_snapshot_across_atomic_replacement(tmp_path, monkeypatch, suffix):
    path = tmp_path / ("params" + suffix)
    new = tmp_path / ("replacement" + suffix)
    for file, value in ((path, 1), (new, 2)):
        with io.open_params(file, "w") as handle:
            handle.write(f"h 0 A {value}\n")
    dimensions = io._parameter_dimensions

    def replace_after_first_pass(handle, tokens):
        shape = dimensions(handle, tokens)
        if new.exists():
            new.replace(path)
        return shape

    monkeypatch.setattr(io, "_parameter_dimensions", replace_after_first_pass)
    assert load_params(path, "AB", "cpu")["bias"][0, 0].item() == 1
    assert load_params(path, "AB", "cpu")["bias"][0, 0].item() == 2


def test_atomic_save_preserves_existing_file_on_invalid_model_or_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "params.dat.gz"
    params = _symmetric_params()
    save_params(path, params, TOKENS)
    original = path.read_bytes()
    invalid = {**params, "bias": torch.full_like(params["bias"], torch.nan)}
    with pytest.raises(ValueError, match="finite"):
        save_params(path, invalid, TOKENS)
    assert path.read_bytes() == original

    @contextmanager
    def failing_output(temporary, *, compressed):
        class Writer:
            def write(self, text):
                handle.write(text[:12])
                raise OSError("simulated write failure")

        with temporary.open("w", encoding="utf-8") as handle:
            yield Writer()

    monkeypatch.setattr(io, "_open_params_output", failing_output)
    with pytest.raises(OutputSerializationError, match="simulated write failure"):
        save_params(path, params, TOKENS)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("numeric", [False, True])
def test_text_saves_preserve_effective_energies(numeric, tmp_path):
    path = tmp_path / "params.dat.gz"
    params = _symmetric_params().copy()
    coupling = params["coupling_matrix"]
    coupling[0, 0, 2, 1] = 2
    coupling[2, 1, 0, 0] = 6
    coupling[1, 2, 1, 2] = 2
    coupling[1, 2, 1, 0] = 7  # Same-site off-diagonal values contribute no energy.
    states = enumerate_states(3, len(TOKENS))
    if numeric:
        io.save_params_oldformat(path, params)
        loaded = io.load_params_oldformat(path, "cpu")
    else:
        save_params(path, params, TOKENS)
        loaded = load_params(path, TOKENS, "cpu")
    torch.testing.assert_close(compute_energy(states, loaded), compute_energy(states, params))
    assert loaded["coupling_matrix"][0, 0, 2, 1].item() == 4


def test_numeric_legacy_writer_with_symmetric_mask_writes_one_triangle(tmp_path):
    params = _symmetric_params()
    mask = params["coupling_matrix"] != 0
    path = tmp_path / "numeric.dat"
    io.save_params_oldformat(path, params, mask)
    assert all(int(row[1]) < int(row[2]) for row in map(str.split, path.read_text().splitlines()) if row[0] == "J")
    for key, value in params.items():
        torch.testing.assert_close(io.load_params_oldformat(path, "cpu")[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(("records", "message"), [
    ("h 0 A 1\n", "Duplicate bias record"),
    ("J 0 1 A B 2\nJ 0 1 A B 3\n", "Duplicate coupling record"),
    ("J 0 1 A B 2\nJ 0 1 B A 3\nJ 0 1 A B 3\n", "Duplicate coupling record"),
    ("unknown 0 1 A B 2\n", "Unknown parameter record"),
])
def test_parser_rejects_invalid_or_repeated_records(tmp_path, monkeypatch, records, message):
    monkeypatch.setattr(io, "_CHUNK_SIZE", 2)
    path = tmp_path / "params.dat"
    path.write_text("h 0 A 0\nh 1 B 0\n" + records, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_params(path, "AB", "cpu")


def test_deprecated_loader_accepts_a_bias_only_file(tmp_path):
    path = tmp_path / "params.dat"
    path.write_text("h 0 A 1\nh 0 B 2\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="load_params_old"):
        params = io.load_params_old(path, "AB", "cpu")
    torch.testing.assert_close(params["bias"], torch.tensor([[1., 2.]]))
    assert not params["coupling_matrix"].any()


@pytest.mark.parametrize(("chains", "expected"), [
    ([0, 1, 0], ("ABA",)),
    (["ABA", "BAB"], ("ABA", "BAB")),
    (torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bfloat16), ("AB",)),
])
def test_save_chains_uses_one_fasta_header_per_decoded_sequence(tmp_path, chains, expected):
    path = tmp_path / "chains.fasta"
    io.save_chains(path, chains, "AB")
    assert path.read_text().splitlines() == [
        line for i, sequence in enumerate(expected) for line in (f">chain_{i}", sequence)
    ]
