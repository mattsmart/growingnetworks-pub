import os
from multiprocessing import cpu_count


# LIBRARY GLOBAL MODS
def set_deep_threads():
    system_threads = cpu_count()
    threads_optimized = {64: "2",    # cluster A - assumes  64 / N tasks per node
                         128: "4"}   # cluster B - assumes 128 / N tasks per node
    if system_threads in threads_optimized.keys():
        threads_int = threads_optimized[system_threads]
        print("init_multiprocessing.py - system_threads=%d is in threads_optimized.keys()" % system_threads)
        print("init_multiprocessing.py - Setting os.environ['OPENBLAS_NUM_THREADS'] (and others) to", threads_int)
        os.environ['MKL_NUM_THREADS'] = threads_int
        os.environ["OMP_NUM_THREADS"] = threads_int
        os.environ["NUMEXPR_NUM_THREADS"] = threads_int
        os.environ["OPENBLAS_NUM_THREADS"] = threads_int
        #os.environ["MKL_THREADING_LAYER"] = "sequential"  # this should be off if NUM_THREADS is not 1
    else:
        print("init_multiprocessing.py - system_threads=%d is not in threads_optimized.keys()" % system_threads)
    return


set_deep_threads()  # this must be set before importing numpy for the first time (during execution)
