import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


def run(dataset: str = "green"):
    # load parquet files
    df_train = read_dataframe(f"{dataset}_tripdata_2021-01.parquet")
    
    df_valid = read_dataframe(f"{dataset}_tripdata_2021-02.parquet")
    
    df_test = read_dataframe(f"{dataset}_tripdata_2021-03.parquet")
    

    # extract the target
    target = 'duration'
    y_train = df_train[target].values
    y_valid = df_valid[target].values
    y_test = df_test[target].values

    # fit the dictvectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_valid, _ = preprocess(df_valid, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # save dictvectorizer and datasets
    dump_pickle(dv, "dv.pkl")
    dump_pickle((X_train, y_train), "train.pkl")
    dump_pickle((X_valid, y_valid), "valid.pkl")
    dump_pickle((X_test, y_test), "test.pkl")


if __name__ == '__main__':

    run()