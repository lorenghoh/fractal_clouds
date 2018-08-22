import ujson as json
import pandas as pd
import pyarrow.parquet as pq

from load_config import config

def find_largest_clouds(filename):
    df = pq.read_pandas(filename).to_pandas()
    return df.cid.value_counts()

if __name__ == '__main__':
    filename = f'{config["tracking"]}/clouds_00000121.pq'
    counts = find_largest_clouds(filename)
    
    counts[:30].to_json('largest_clouds.json')