from time import time


def timer(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        res = func(*args, **kwargs)
        t2 = time()
        print("{0:<20}{1:.3f}s".format("EXECUTION TIME:", t2 - t1))
        return res
    return wrapper
