"""Command-line adapter for alignment conversion and preprocessing."""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert and preprocess FASTA or Stockholm alignments."
    )
    parser.add_argument("input", help="Input FASTA or Stockholm alignment.")
    parser.add_argument("-o", "--output", required=True, help="Output FASTA path.")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "fasta", "stockholm"],
        help="Input format. By default it is detected from the file content.",
    )
    parser.add_argument(
        "--alignment-index",
        type=int,
        default=None,
        help="Alignment to select when a Stockholm file contains multiple alignments.",
    )
    parser.add_argument(
        "--remove-insertions",
        action="store_true",
        help="Remove dots and lowercase insertion residues.",
    )
    parser.add_argument(
        "--max-gap-fraction",
        type=float,
        default=None,
        help="Remove sequences with a gap fraction greater than this value.",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Keep only the first occurrence of each processed sequence.",
    )
    parser.add_argument(
        "--alphabet",
        default=None,
        help="Optionally validate against protein, RNA, DNA, or custom tokens.",
    )
    parser.add_argument(
        "--gap-token",
        default="-",
        help="Single character counted as a gap (default: '-').",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=0,
        help="Wrap FASTA sequences at this width; 0 writes one line per sequence.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for a JSON preprocessing report.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()

    from adabmDCA.preprocessing import preprocess_alignment

    result = preprocess_alignment(
        args.input,
        output_path=args.output,
        input_format=args.input_format,
        alignment_index=args.alignment_index,
        remove_insertions=args.remove_insertions,
        max_gap_fraction=args.max_gap_fraction,
        remove_duplicates=args.remove_duplicates,
        alphabet=args.alphabet,
        gap_token=args.gap_token,
        line_width=args.line_width,
    )
    if args.report is not None:
        result.report.to_json(args.report)

    report = result.report
    print("Alignment preprocessing completed successfully")
    print(f"  Input sequences:  {report.input_sequences}")
    print(f"  Output sequences: {report.output_sequences}")
    print(f"  Input length:     {report.input_length}")
    print(f"  Output length:    {report.output_length}")
    print(f"  Insertions removed: {report.removed_insertion_characters}")
    print(f"  Gaps normalized:    {report.normalized_gap_characters}")
    print(f"  Gap-filtered:       {report.removed_for_gap_fraction}")
    print(f"  Duplicates removed: {report.removed_as_duplicates}")
    print(f"  Output: {result.output_path}")
    if args.report is not None:
        print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
