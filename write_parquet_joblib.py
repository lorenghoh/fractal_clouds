import glob, h5py

import dask.dataframe as dd

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from joblib import Parallel, delayed

keys = {
        "condensed": 0,
        "condensed_edge": 1,
        "condensed_env": 2,
        "condensed_shell": 3,
        "core": 4,
        "core_edge": 5,
        "core_env": 6,
        "core_shell": 7,
        "plume": 8,
        }

_i = int
def write_parquet(time, file):
    rec = {'cid': [], 'type': [], 'coord': []}

    def append_items(type, obj):
        for index in obj:
            rec['cid'].append(_i(obj.name.split('/')[1]))
            rec['type'].append(keys[type])
            rec['coord'].append(index)

    with h5py.File(file, 'r', libver='latest') as h5_file:
        for cid in h5_file.keys():
            if _i(cid) == -1: continue # Ignore noise
            h5_file[cid].visititems(append_items)

    df = pd.DataFrame.from_dict(rec)
    loc = '/scratchSSD/loh/tracking/CGILS_301K'
    pq.write_table(pa.Table.from_pandas(df), 
                   f'{loc}/clouds_{time:08d}.pq',
                   use_dictionary=True)

if __name__ == '__main__':
    loc = '/newtera/loh/workspace/loh_tracker/hdf5'
    filelist = sorted(glob.glob(f'{loc}/clouds_*.h5'))

    Parallel(n_jobs=16)(delayed(write_parquet)(time, file) 
                        for time, file in enumerate(filelist))