# src/dlc_live_torch/__main__.py

import sys
from .app import main

if __name__ == "__main__":
    # The 'if __name__ == "__main__"' block from the original app.py is now here
    # to ensure multiprocessing works correctly when the package is run as a script.
    main()
