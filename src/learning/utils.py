"""
file to hold utility functions

"""


import time

def timed(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{(t1 - t0):.4f} seconds")

        return (result, t1 - t0)
    return wrapper

 

