#!/usr/bin/env python
"""
Wrapper script to download ScanNet scenes non-interactively
Handles the TOS agreement and .sens file prompts automatically
"""
import subprocess
import sys
import os
import select
import fcntl

def download_scene(download_script, out_dir, scene_id, file_type='.sens', skip_existing=False):
    """Download a single ScanNet scene, handling interactive prompts"""
    
    cmd = [
        'python3', download_script,
        '-o', out_dir,
        '--id', scene_id,
        '--type', file_type
    ]
    
    if skip_existing:
        cmd.append('--skip_existing')
    
    # The script prompts:
    # 1. TOS agreement: "Press any key to continue"
    # 2. For v2 files: "Press 'n' to exclude downloading .sens files" (for .sens) or similar for other types
    # We want to answer: 'y' (or any key) for TOS, and 'y' (or any key) for the file type
    
    # Use pexpect if available (best option)
    try:
        import pexpect
        child = pexpect.spawn(' '.join(cmd), encoding='utf-8', timeout=3600)
        
        # Handle TOS prompt
        child.expect(['Press any key', 'Press.*key'], timeout=30)
        child.sendline('y')
        
        # Handle file type prompt (if it appears for v2)
        # For non-.sens file types, the script may exit immediately after download
        # For .txt files, there might be a different prompt or no prompt
        try:
            index = child.expect(['Press.*n.*exclude', 'Press.*exclude', 'Note: ScanNet v2', 'Press any key'], timeout=5)
            # Got a prompt, send response
            child.sendline('y')  # We want the files (press any key except 'n')
            # Wait for process to complete
            child.expect(pexpect.EOF, timeout=3600)
        except pexpect.TIMEOUT:
            # No prompt appeared, wait for process to complete
            child.expect(pexpect.EOF, timeout=3600)
        except pexpect.EOF:
            # Process already ended - this is OK for non-.sens file types
            # The download may have completed successfully
            pass
        
        output = child.before
        child.close()
        
        # Check if download was successful by looking at exit status or output
        # Exit status might be None if process ended normally, so also check output
        success = (child.exitstatus == 0) or (child.exitstatus is None and 'Downloaded scan' in output)
        
        return success, output
        
    except ImportError:
        # Fallback: use subprocess with pre-written input
        # Note: This may not work perfectly if prompts appear at unexpected times
        input_data = 'y\ny\n'  # TOS agreement, .sens confirmation
        
        # Use universal_newlines for Python < 3.7 compatibility, text=True for >= 3.7
        import sys
        if sys.version_info >= (3, 7):
            text_arg = {'text': True}
        else:
            text_arg = {'universal_newlines': True}
        
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **text_arg
        )
        
        stdout, _ = proc.communicate(input=input_data, timeout=3600)
        return proc.returncode == 0, stdout


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: download_scannet_wrapper.py <download_script> <out_dir> <scene_id> [--type TYPE] [--skip_existing]")
        sys.exit(1)
    
    download_script = sys.argv[1]
    out_dir = sys.argv[2]
    scene_id = sys.argv[3]
    
    # Parse optional arguments
    file_type = '.sens'  # default
    skip_existing = False
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == '--type' and i + 1 < len(sys.argv):
            file_type = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--skip_existing':
            skip_existing = True
            i += 1
        else:
            i += 1
    
    success, output = download_scene(download_script, out_dir, scene_id, file_type, skip_existing)
    
    print(output)
    sys.exit(0 if success else 1)
