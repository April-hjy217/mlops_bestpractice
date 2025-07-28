#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime) \
                        .dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype(int).astype(str)
    return df

def main():

    year, month = 2023, 1
    path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/' \
           f'yellow_tripdata_{year:04d}-{month:02d}.parquet'
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(path, categorical)

    dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)

    y = df['duration'].values


    lr = LinearRegression()
    lr.fit(X, y)

    with open('model.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)

    print("✅ model.bin")

if __name__ == '__main__':
    main()
