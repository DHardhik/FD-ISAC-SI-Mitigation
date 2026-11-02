"""Project-root conftest: makes `import config, channels, ...` work when running `python -m pytest` from anywhere."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
