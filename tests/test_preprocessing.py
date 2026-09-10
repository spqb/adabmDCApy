import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


STOCKHOLM = """# STOCKHOLM 1.0
s1 ACd.-E
s2 ACd.DE
s3 A.aCDE
//
"""


class AlignmentPreprocessingTests(unittest.TestCase):
    def test_remove_insertions_deletes_dots_and_lowercase(self):
        from adabmDCA import Alignment, remove_insertions

        alignment = Alignment(("s1", "s2"), ("ACd.-E", "A.aC-E"))
        cleaned = remove_insertions(alignment)

        self.assertEqual(cleaned.sequences, ("AC-E", "AC-E"))
        self.assertEqual(cleaned.names, alignment.names)

    def test_remove_insertions_revalidates_alignment_length(self):
        from adabmDCA import Alignment, AlignmentLengthError, remove_insertions

        alignment = Alignment(("s1", "s2"), ("AaB", "ABC"))
        with self.assertRaises(AlignmentLengthError):
            remove_insertions(alignment)

    def test_gap_filter_retains_exact_threshold(self):
        from adabmDCA import Alignment, filter_gap_fraction

        alignment = Alignment(("exact", "above", "clean"), ("ABC-", "AB--", "ABCD"))
        result = filter_gap_fraction(alignment, max_gap_fraction=0.25)

        self.assertEqual(result.alignment.names, ("exact", "clean"))
        self.assertEqual(result.keep_mask, (True, False, True))
        self.assertEqual(result.gap_fractions, (0.25, 0.5, 0.0))
        self.assertEqual(result.removed_names, ("above",))

    def test_stockholm_conversion_uses_fasta_gap_symbol(self):
        from adabmDCA import convert_stockholm_to_fasta, read_alignment

        with TemporaryDirectory() as directory:
            source = Path(directory) / "family.sto"
            output = Path(directory) / "family.fasta"
            source.write_text(STOCKHOLM, encoding="utf-8")

            conversion = convert_stockholm_to_fasta(source, output)
            round_trip = read_alignment(output)
            written = output.read_text(encoding="utf-8")

        self.assertEqual(conversion.input_format, "stockholm")
        self.assertEqual(round_trip.names, ("s1", "s2", "s3"))
        self.assertEqual(round_trip.sequences, ("ACd--E", "ACd-DE", "A-aCDE"))
        self.assertNotIn(".", written)

    def test_preprocessing_normalizes_dots_before_gap_filtering(self):
        from adabmDCA import preprocess_alignment

        alignment = """>exact\nABC.\n>above\nAB..\n>clean\nABCD\n"""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.fasta"
            output = Path(directory) / "clean.fasta"
            source.write_text(alignment, encoding="utf-8")
            result = preprocess_alignment(
                source,
                output_path=output,
                max_gap_fraction=0.25,
            )

            written = output.read_text(encoding="utf-8")

        self.assertEqual(result.alignment.names, ("exact", "clean"))
        self.assertEqual(result.alignment.sequences, ("ABC-", "ABCD"))
        self.assertEqual(result.report.normalized_gap_characters, 3)
        self.assertEqual(result.report.removed_names, ("above",))
        self.assertNotIn(".", written)

    def test_pipeline_reports_gap_and_duplicate_removal(self):
        from adabmDCA import preprocess_alignment

        with TemporaryDirectory() as directory:
            source = Path(directory) / "family.sto"
            output = Path(directory) / "clean.fasta"
            report_path = Path(directory) / "report.json"
            source.write_text(STOCKHOLM, encoding="utf-8")

            result = preprocess_alignment(
                source,
                output_path=output,
                remove_insertions=True,
                max_gap_fraction=0.2,
                remove_duplicates=True,
                alphabet="ACDE-",
            )
            result.report.to_json(report_path)
            report_json = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.alignment.names, ("s2",))
        self.assertEqual(result.alignment.sequences, ("ACDE",))
        self.assertEqual(result.keep_mask, (False, True, False))
        self.assertEqual(result.report.removed_insertion_characters, 6)
        self.assertEqual(result.report.removed_for_gap_fraction, 1)
        self.assertEqual(result.report.removed_as_duplicates, 1)
        self.assertEqual(result.report.removed_names, ("s1", "s3"))
        self.assertEqual(report_json["output_sequences"], 1)

    def test_multiple_stockholm_alignments_require_selection(self):
        from adabmDCA import AlignmentFormatError, read_alignment

        multiple = STOCKHOLM + STOCKHOLM.replace("s1", "x1").replace("s2", "x2").replace("s3", "x3")
        with TemporaryDirectory() as directory:
            source = Path(directory) / "multiple.sto"
            source.write_text(multiple, encoding="utf-8")
            with self.assertRaises(AlignmentFormatError):
                read_alignment(source)
            selected = read_alignment(source, alignment_index=1)

        self.assertEqual(selected.names, ("x1", "x2", "x3"))

    def test_interleaved_stockholm_fragments_are_concatenated(self):
        from adabmDCA import read_alignment

        interleaved = """# STOCKHOLM 1.0
s1 AC
s2 AG
#=GC RF xx

s1 DE
s2 D-
//
"""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "interleaved.sto"
            source.write_text(interleaved, encoding="utf-8")
            alignment = read_alignment(source)

        self.assertEqual(alignment.sequences, ("ACDE", "AGD-"))

    def test_processing_config_is_supported(self):
        from adabmDCA import AlignmentProcessingConfig, preprocess_alignment

        with TemporaryDirectory() as directory:
            source = Path(directory) / "family.sto"
            source.write_text(STOCKHOLM, encoding="utf-8")
            result = preprocess_alignment(
                source,
                config=AlignmentProcessingConfig(
                    remove_insertions=True,
                    max_gap_fraction=0.25,
                    remove_duplicates=True,
                ),
            )

        self.assertEqual(result.alignment.names, ("s1", "s2"))
        self.assertEqual(result.keep_mask, (True, True, False))

    def test_preprocess_cli_writes_fasta_and_json_report(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "family.sto"
            output = Path(directory) / "clean.fasta"
            report = Path(directory) / "report.json"
            source.write_text(STOCKHOLM, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "adabmDCA.cli",
                    "preprocess",
                    str(source),
                    "--output",
                    str(output),
                    "--remove-insertions",
                    "--max-gap-fraction",
                    "0.2",
                    "--remove-duplicates",
                    "--report",
                    str(report),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(report.is_file())
            self.assertIn("Output sequences: 1", completed.stdout)
            self.assertNotIn(".", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
