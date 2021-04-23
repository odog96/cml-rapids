## Feature Engineering using dask

import time

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cudf as dd
import dask_cudf as dc
from feature_engineering_adv import feature_engineering
import gc

cluster = LocalCUDACluster()
client = Client(cluster)
client

### Load Data
bureau_balance = dc.read_parquet('raw_data/bureau_balance.parquet')
bureau = dc.read_parquet('raw_data/bureau.parquet')
cc_balance = dc.read_parquet('raw_data/cc_balance.parquet')
payments = dc.read_parquet('raw_data/payments.parquet')
pc_balance = dc.read_parquet('raw_data/pc_balance.parquet')
prev = dc.read_parquet('raw_data/prev.parquet')
train = dd.read_parquet('raw_data/train.parquet')
test = dd.read_parquet('raw_data/test.parquet')

train_target = dc.from_cudf(train['TARGET'], npartitions=1)
unified = dd.concat([train.drop('TARGET', axis=1), test])

unified_dask = dc.from_cudf(unified, npartitions=2)
del(unified)
gc.collect()

avg_bureau, sum_cc_balance, sum_payments, \
    sum_pc_balance, sum_prev, unified_feat = feature_engineering(bureau_balance,
        bureau, cc_balance, payments, pc_balance,
        prev, unified_dask)

train_rows = train.shape
train_feats = unified_feat.iloc[:307511].merge(train_target, how='left', 
                                           left_index=True, right_index=True)
test_feats = unified_feat.iloc[307511:]

avg_bureau.to_parquet(path='data_eng/avg_bureau')
sum_cc_balance.to_parquet(path='data_eng/sum_cc_balance')
sum_payments.to_parquet(path='data_eng/sum_payments')
sum_pc_balance.to_parquet(path='data_eng/sum_pc_balance')
sum_prev.to_parquet(path='data_eng/sum_prev')

train_feats.to_parquet('data_eng/feats/train_feats.parquet')
test_feats.to_parquet('data_eng/feats/test_feats.parquet')

client.close()