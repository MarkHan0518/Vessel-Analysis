"""
This module contains the code to find out the number of CPU cores and threads in your computer.
"""


import os
import multiprocessing


num_threads = os.cpu_count()
print(f"Number of available threads: {num_threads}")

num_cores = multiprocessing.cpu_count()
print(f"Number of available CPU cores: {num_cores}")