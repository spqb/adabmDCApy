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
