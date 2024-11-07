import subprocess
import sys
import os

def main():
    script_path = os.path.join(os.path.dirname(__file__), 'adabmDCA.sh')
    subprocess.run([script_path] + sys.argv[1:], check=True)

if __name__ == "__main__":
    main()