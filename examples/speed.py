import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'
import timeit

import pandas as pd

from lightbt import warmup

_ = os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    print('warmup:', warmup())
    print('warmup:', warmup())
    print(timeit.timeit('warmup()', number=1000, globals=locals()))
