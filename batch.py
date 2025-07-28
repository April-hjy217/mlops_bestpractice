#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")

def get_input_path(year, month):
    default = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )
    pattern = os.getenv("INPUT_FILE_PATTERN", default)
    return pattern.format(year=year, month=month)


def get_output_path(year, month):
    default = f"output/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    pattern = os.getenv("OUTPUT_FILE_PATTERN", default)
    return pattern.format(year=year, month=month)


def get_output_path(year, month):
    default = (
        "s3://nyc-duration-prediction-alexey/"
        "taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    )
    pattern = os.getenv("OUTPUT_FILE_PATTERN", default)
    return pattern.format(year=year, month=month)

def prepare_data(df, categorical):
    df['duration'] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = (
        df[categorical]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    return df


def read_data(filename, categorical):
    if filename.startswith("s3://") and S3_ENDPOINT_URL:
        storage_options = {
            "client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}
        }
        df = pd.read_parquet(
            filename,
            storage_options=storage_options
        )
    else:
        df = pd.read_parquet(filename)

    return prepare_data(df, categorical)

def main(year, month):

    os.makedirs('output', exist_ok=True)


    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)


    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print('predicted mean duration:', y_pred.mean())

 
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    

    
    storage_options = {}
    if os.getenv("S3_ENDPOINT_URL"):
        storage_options = {
          "client_kwargs": {"endpoint_url": os.getenv("S3_ENDPOINT_URL")}
        }

    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        index=False,
        storage_options=storage_options
    )


    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python batch.py <year> <month>")
        sys.exit(1)
    year, month = map(int, sys.argv[1:])
    main(year, month)
