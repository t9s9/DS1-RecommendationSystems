from time import time

import pandas as pd
import streamlit as st


def timer(func):
    """ Decorator to measure function execution time """

    def wrapper(*args, **kwargs):
        t1 = time()
        res = func(*args, **kwargs)
        t2 = time()
        print("{0:<20}{1:.3f}s".format(f"EXECUTION TIME OF {func.__module__}: ", t2 - t1))
        return res

    return wrapper


def force_rerun():
    """ Forces streamlit to rerun the script: see https://github.com/streamlit/streamlit/issues/653 """
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))


@st.cache
def read_csv_cached(path):
    return pd.read_csv(path)
