"""Alphabet definitions shared by lightweight and tensor-based APIs."""

TOKENS_PROTEIN = "-ACDEFGHIKLMNPQRSTVWY"
TOKENS_RNA = "-ACGU"
TOKENS_DNA = "-ACGT"


def get_tokens(alphabet: str) -> str:
    """Return built-in tokens or an explicitly supplied custom alphabet."""
    if not isinstance(alphabet, str):
        raise TypeError("Argument 'alphabet' must be of type str")
    if alphabet == "protein":
        return TOKENS_PROTEIN
    if alphabet == "rna":
        return TOKENS_RNA
    if alphabet == "dna":
        return TOKENS_DNA
    return alphabet


def detect_alphabet(sequences) -> str:
    """Select the smallest compatible standard alphabet (DNA wins ties).

    All symbols must fit a standard alphabet; custom tokens are never inferred.
    Gap-only or empty input cannot identify an alphabet.
    """
    from adabmDCA.api.exceptions import InputValidationError

    symbols = set()
    for sequence in sequences:
        symbols.update(sequence)
    if symbols - {"-"}:
        for name in ("dna", "rna", "protein"):
            if symbols <= set(get_tokens(name)):
                return name
    raise InputValidationError(
        "Cannot automatically detect a standard alphabet compatible with the data. "
        "Specify --alphabet protein, dna, rna, or an explicit custom alphabet.",
        details={"observed_tokens": sorted(symbols)},
    )
