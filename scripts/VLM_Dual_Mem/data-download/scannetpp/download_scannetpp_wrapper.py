#!/usr/bin/env python3
"""
Wrapper script to run download_scannetpp.py non-interactively
This script automatically confirms the download prompt
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: download_scannetpp_wrapper.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Check if config file exists
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    # Get the path to the download script
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_scannetpp.py"
    
    if not download_script.exists():
        print(f"Error: Download script not found: {download_script}")
        sys.exit(1)
    
    print("=" * 60)
    print("ScanNet++ Download Wrapper")
    print("=" * 60)
    print(f"Config file: {config_file}")
    print(f"Download script: {download_script}")
    print("=" * 60)
    print("")
    print("Note: This wrapper will automatically confirm the download.")
    print("The download may require ~1.5TB of disk space.")
    print("")
    
    # Run the download script with 'y' piped to stdin
    try:
        # Use subprocess to run the script and pipe 'y' to stdin
        result = subprocess.run(
            [sys.executable, str(download_script), config_file],
            input='y\n',
            text=True,
            check=False
        )
        
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running download script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
