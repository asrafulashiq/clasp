from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

a = Path('detect.py')
print(a.exists())
print(os.getcwd())
print(os.path.dirname(os.path.abspath(__file__)))


