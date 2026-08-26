import sys
import importlib
from contextlib import contextmanager

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


def main():
    print(f"🧬 adabmDCA version: {importlib.import_module('adabmDCA').__version__}")

    if len(sys.argv) < 2:
        print(f"Error: No command provided. {_usage()}")
        sys.exit(1)

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Error: Invalid command '{command}'. {_usage()}")
        sys.exit(1)

    module = importlib.import_module(module_name)
    with script_argv(command, sys.argv[2:]):
        module.main()

if __name__ == "__main__":
    main()
