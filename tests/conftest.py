import os
import sys

# macOS: torch and faiss-cpu each ship their own libomp. Loading both into one process aborts
# ("OMP: Error #15"), and allowing the duplicate runtime alone deadlocks FAISS's threads. Only the
# test session loads both (the torch tests next to the FAISS tests once ".[dev,clm]" is installed;
# toolrank serve runs the heads in numpy, without torch), so the test session allows the duplicate
# runtime and keeps OpenMP to one thread.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
