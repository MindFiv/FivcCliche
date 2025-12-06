#!/usr/bin/env python3
"""
Ruff linter wrapper that converts relative paths to absolute paths.
This makes error messages clickable in IDEs like VS Code and PyCharm.
"""

import subprocess
import sys
import re
from pathlib import Path

def convert_to_absolute_paths(output: str, project_root: Path) -> str:
    """Convert relative file paths in Ruff output to absolute paths."""
    # Pattern 1: --> relative/path/file.py:line:col (full format)
    pattern1 = r'-->\s+([^:\s]+):(\d+):(\d+)'

    # Pattern 2: relative/path/file.py:line: (pylint format)
    pattern2 = r'^([^:\s]+):(\d+):'

    def replace_path1(match):
        rel_path = match.group(1)
        line = match.group(2)
        col = match.group(3)
        abs_path = (project_root / rel_path).resolve()
        return f'--> {abs_path}:{line}:{col}'

    def replace_path2(match):
        rel_path = match.group(1)
        line = match.group(2)
        abs_path = (project_root / rel_path).resolve()
        return f'{abs_path}:{line}:'

    output = re.sub(pattern1, replace_path1, output)
    output = re.sub(pattern2, replace_path2, output, flags=re.MULTILINE)
    return output

def main():
    project_root = Path(__file__).parent.parent
    
    # Run ruff check with the provided arguments
    cmd = [sys.executable, '-m', 'ruff', 'check'] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        # Convert relative paths to absolute paths
        converted_output = convert_to_absolute_paths(output, project_root)
        
        # Print the converted output
        print(converted_output, end='')
        
        # Exit with the same code as ruff
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running ruff: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

