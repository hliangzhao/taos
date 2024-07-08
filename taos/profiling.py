"""
Profiling the algorithms' computation overhead.
"""
import functools
import time


def profiling(algo_name):
    """
    A decorator to profile the computation time of the given algorithm.
    """

    def decorator(algo):
        @functools.wraps(algo)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = algo(*args, **kwargs)
            end = time.time()

            print("[{0}] computation overhead: {1:.6f} secs".format(algo_name, end - start))
            return res

        return wrapper

    return decorator
