"""Standard-only alphabet inference for command-line workflows."""
from argparse import Namespace

import pytest

from adabmDCA.alphabet import detect_alphabet
from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.scripts._frontend import resolve_alphabet


@pytest.mark.parametrize(('sequences', 'expected'), [
    (['ACGT-', 'TGCA-'], 'dna'),
    (['ACGU-'], 'rna'),
    (['ACDEFGHIKLMNPQRSTVWY-'], 'protein'),
    (['ACG-'], 'dna'),
])
def test_standard_detection(sequences, expected):
    assert detect_alphabet(sequences) == expected


@pytest.mark.parametrize('sequences', [['ACGX'], ['ACGTU'], ['01'], ['---'], []])
def test_requires_explicit_alphabet(sequences):
    with pytest.raises(InputValidationError, match='explicit custom alphabet'):
        detect_alphabet(sequences)


def test_model_disambiguates_subset_alignment(tmp_path):
    data = tmp_path / 'data.fa'
    data.write_text('>s\nACG\n')
    params = tmp_path / 'params.dat'
    params.write_text('h 0 A 0\nh 0 E 0\n')
    args = Namespace(alphabet='auto', data=data, path_params=params)
    resolve_alphabet(args)
    assert args.alphabet == 'protein'


def test_model_only_rna(tmp_path):
    params = tmp_path / 'params.dat'
    params.write_text('J 0 1 A U 0\nh 0 - 0\n')
    args = Namespace(alphabet='auto', path_params=params)
    resolve_alphabet(args)
    assert args.alphabet == 'rna'


def test_explicit_custom_does_not_inspect_inputs():
    args = Namespace(alphabet='01', data='missing.fa')
    resolve_alphabet(args)
    assert args.alphabet == '01'


def test_cli_training_defaults_to_auto():
    from adabmDCA.scripts.train import create_parser
    assert create_parser().parse_args(['-d', 'data.fa']).alphabet == 'auto'
