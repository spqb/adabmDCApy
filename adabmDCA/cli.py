import subprocess
import sys
import os

def main():
    # Get the directory of the current script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Check if the first positional argument is provided
    if len(sys.argv) < 2:
        print("Error: No command provided. Use 'train', 'sample', 'contacts', 'energies', or 'DMS'.")
        sys.exit(1)

    # Assign the first positional argument to a variable
    COMMAND = sys.argv[1]

    # Map the command to the corresponding script
    match COMMAND:
        case "train":
            SCRIPT = "train.py"
        case "sample":
            SCRIPT = "sample.py"
        case "contacts":
            SCRIPT = "contacts.py"
        case "energies":
            SCRIPT = "energies.py"
        case "DMS":
            SCRIPT = "dms.py"
        case _:
            print(f"Error: Invalid command '{COMMAND}'. Use 'train', 'sample', 'contacts', 'energies', or 'DMS'.")
            sys.exit(1)

    # Run the corresponding Python script with the remaining optional arguments
    REPO_SCRIPTS = os.path.join(SCRIPT_DIR, "scripts")
    script_path = os.path.join(REPO_SCRIPTS, SCRIPT)
    proc = subprocess.call(
        [sys.executable, script_path] + sys.argv[2:],
    )

if __name__ == "__main__":
    main()