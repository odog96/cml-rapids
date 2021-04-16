## build Model dask_cudf


import time

import dask.dataframe
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf
import numpy as np
import cudf

cluster = LocalCUDACluster()
client = Client(cluster)
client

## reload our parquet
avg_bureau = dask_cudf.read_parquet('data_eng/avg_bureau.parq')
sum_cc_balance = dask_cudf.read_parquet(path='data_eng/sum_cc_balance.parq')
sum_payments = dask_cudf.read_parquet(path='data_eng/sum_payments.parq')
sum_pc_balance = dask_cudf.read_parquet(path='data_eng/sum_pc_balance.parq')
sum_prev = dask_cudf.read_parquet(path='data_eng/sum_prev.parq')
train = dask_cudf.read_csv('data/application_train.csv')

## Lets try and merge everything with dask this time
train_feat = train.drop('TARGET', axis=1) \
    .merge(avg_bureau, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR') \
    .merge(sum_cc_balance, how='left', left_on='SK_ID_CURR', right_index=True) \
    .merge(sum_payments, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR') #\
    #.merge(sum_pc_balance, how='left', left_on='SK_ID_CURR', right_index=True)

train_feat.to_parquet(path='data_eng/train_feat')


train_feat.head(15)