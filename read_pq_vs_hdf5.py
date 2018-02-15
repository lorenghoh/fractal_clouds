import h5py, time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def get_timing(f, path, niter):
    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    for i in range(niter):
        f(path)
    elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - start

    return elapsed

# Naive, full memory-mapping operation (150x faster)
def read_pyarrow(path):
    return pq.read_table(f'{path}/clouds.pq', nthreads=6).to_pandas()

def read_hdf5(path):
    def read_cols(type, obj):
        return list(obj)

    with h5py.File(f'{path}/clouds.h5') as h5_file:
        for cid in h5_file.keys():
            h5_file[cid].visititems(read_cols)

# Accessing a single cid entry (#8230)
def read_pyarrow_cid(path):
    df = pq.read_table(f'{path}/clouds.pq', nthreads=6).to_pandas()
    return df[df.cid == 8230]

def read_hdf5_cid(path):
    def read_cols(type, obj):
        return list(obj)

    with h5py.File(f'{path}/clouds.h5') as h5_file:
        h5_file['8230'].visititems(read_cols)

path = '/nodessd/loh/temp'
readers = [
    ('pyarrow', lambda path: read_pyarrow_cid(path)),
    ('hdf5', lambda path: read_hdf5_cid(path))
]

NITER = 10
for reader_name, f in readers:
    elapsed = get_timing(f, path, NITER) / NITER
    result = elapsed
    print(reader_name, result)
