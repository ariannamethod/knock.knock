#!/usr/bin/env python3
# talk2haze.py — Simple bridge to HAZE REPL
#
# Direct connection to haze interactive mode.
# No routing, no CLOUD, just pure HAZE conversation.

import sys
from pathlib import Path

# Add haze directory to path
sys.path.insert(0, str(Path(__file__).parent / "haze"))

# Import and run HAZE REPL
from haze import run

if __name__ == "__main__":
    run.main()
