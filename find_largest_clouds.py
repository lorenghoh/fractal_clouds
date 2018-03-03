import pandas as pd
import pyarrow.parquet as pq 

def find_largest_clouds(filename):
    df = pq.read_pandas(filename).to_pandas()
    return df.cid.value_counts()

if __name__ == '__main__':
    filename = 'tracking/clouds_00000121.pq'
    counts = find_largest_clouds(filename)

    print(counts[:10])
