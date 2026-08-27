import importlib
import sys
from contextlib import contextmanager

from adabmDCA.api.exceptions import AdabmDCAError

COMMANDS = {
    "train": "adabmDCA.scripts.train",
    "sample": "adabmDCA.scripts.sample",
    "contacts": "adabmDCA.scripts.contacts",
    "energies": "adabmDCA.scripts.energies",
    "dms": "adabmDCA.scripts.dms",
    "DMS": "adabmDCA.scripts.dms",
    "entropy": "adabmDCA.scripts.td_integration",
    "reintegrate": "adabmDCA.scripts.reintegrate",
    "profmark": "adabmDCA.scripts.profmark",
    "plot-training-log": "adabmDCA.plot_training_log",
    "plot_training_log": "adabmDCA.plot_training_log",
    "preprocess": "adabmDCA.scripts.preprocess",
}

COMMAND_DESCRIPTIONS = {
    "train": "Train a DCA model.",
    "sample": "Generate sequences from a trained model.",
    "contacts": "Predict residue-residue contacts.",
    "energies": "Compute DCA energies for aligned sequences.",
    "dms": "Score all single-residue substitutions.",
    "entropy": "Estimate sequence entropy by thermodynamic integration.",
    "reintegrate": "Reintegrate previously removed alignment positions.",
    "profmark": "Create profile-model training and test splits.",
    "plot-training-log": "Plot metrics from a training log.",
    "preprocess": "Convert and preprocess sequence alignments.",
}


@contextmanager
def script_argv(command: str, args: list[str]):
    previous_argv = sys.argv[:]
    sys.argv = [f"adabmDCA {command}", *args]
    try:
        yield
    finally:
        sys.argv = previous_argv


def _usage() -> str:
    commands = "', '".join(COMMANDS)
    return f"Use one of '{commands}'."


def _help() -> str:
    command_lines = "\n".join(f"  {command:<20} {description}" for command, description in COMMAND_DESCRIPTIONS.items())
    return (
        "usage: adabmDCA <command> [options]\n\n"
        "Direct Coupling Analysis in Python.\n\n"
        "commands:\n"
        f"{command_lines}\n\n"
        "Run 'adabmDCA <command> --help' for command-specific options."
    )


def render_error(error: AdabmDCAError) -> str:
    """Render a structured application error without exposing a traceback."""
    lines = [f"Error [{error.code}]: {error.message}"]
    lines.extend(f"  {name}: {value}" for name, value in error.details.items())
    return "\n".join(lines)


def main() -> int:
    print(f"🧬 adabmDCA version: {importlib.import_module('adabmDCA').__version__}")

    if len(sys.argv) < 2:
        print(f"Error: No command provided. {_usage()}")
        return 2

    command = sys.argv[1]
    if command in {"-h", "--help", "help"}:
        print(_help())
        return 0
    if command in {"-V", "--version"}:
        return 0

    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Error: Invalid command '{command}'. {_usage()}")
        return 2

    module = importlib.import_module(module_name)
    try:
        with script_argv(command, sys.argv[2:]):
            status = module.main()
    except KeyboardInterrupt:
        print("Error: operation interrupted.", file=sys.stderr)
        return 130
    except AdabmDCAError as exc:
        print(render_error(exc), file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return status if isinstance(status, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
