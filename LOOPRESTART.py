import subprocess
import sys

while True:
    try:
        subprocess.check_call(['python', 'main.py', '--output-directory', 'D:/SDTEMP/gens','--temp-directory','D:/SDTEMP/trash', "--use-pytorch-cross-attention", "--input-directory", "D:/SDTEMP/input",'--disable-xformers'])
    except KeyboardInterrupt:
        print("\nInterrupted... restarting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
