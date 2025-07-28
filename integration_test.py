import os
import pandas as pd
from datetime import datetime
import subprocess
from pyarrow import fs, parquet
import pytest

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1,    1,    dt(1, 2), dt(1, 10)),
    (1, None,    dt(1, 2, 0), dt(1, 2, 59)),
    (3,    4,    dt(1, 2, 0), dt(2, 2, 1)),
]
cols = ['PULocationID','DOLocationID','tpep_pickup_datetime','tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=cols)


options = {'client_kwargs': {'endpoint_url': os.getenv('S3_ENDPOINT_URL')}}
input_path = os.getenv('INPUT_FILE_PATTERN').format(year=2023, month=1)

df_input.to_parquet(
    input_path, 
    engine='pyarrow', 
    compression=None,
    index=False,
    storage_options=options
)

subprocess.run(["python", "batch.py", "2023", "1"], check=True)

output_path = os.getenv("OUTPUT_FILE_PATTERN").format(year=2023, month=1)
s3 = fs.S3FileSystem(endpoint_override=os.getenv("S3_ENDPOINT_URL"))

path_no_scheme = output_path.replace("s3://", "")
table = parquet.read_table(path_no_scheme, filesystem=s3)
df_out = table.to_pandas()

total = df_out["predicted_duration"].sum()
print("Sum of predicted_duration:", total) 
assert pytest.approx(36.28, rel=1e-2) == total




