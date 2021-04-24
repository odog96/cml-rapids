## Feature Engineering using dask
## This is currently broken and doesn't run
## I think we need a different way to trigger dask?

from dask_cuda import LocalCUDACluster
import time
from dask.distributed import Client
import cudf as dd
import dask_cudf as dc
from feature_engineering_adv import feature_engineering
import gc

print("Start Cluster")
cluster = LocalCUDACluster(n_workers=2, 
                        threads_per_worker=6, 
                        protocol="ucx", 
                        enable_tcp_over_ucx=True, 
                        enable_nvlink=True,
                        rmm_pool_size="6GB")

time.sleep(10)

print("create client")


#cluster
client = Client()
#client

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

train_index = train.index.to_arrow().tolist()
test_index = test.index.to_arrow().tolist()

del(train)
del(test)
del(unified)
gc.collect()

unified_feat = feature_engineering(bureau_balance,
        bureau, cc_balance, payments, pc_balance,
        prev, unified_dask, dc, checks=False)

train_feats = unified_feat.loc[train_index].merge(train_target, how='left', 
                                           left_index=True, right_index=True)
test_feats = unified_feat.loc[test_index]

#avg_bureau.to_parquet(path='data_eng/avg_bureau')
#sum_cc_balance.to_parquet(path='data_eng/sum_cc_balance')
#sum_payments.to_parquet(path='data_eng/sum_payments')
#sum_pc_balance.to_parquet(path='data_eng/sum_pc_balance')
#sum_prev.to_parquet(path='data_eng/sum_prev')

train_feats.to_parquet('data_eng/feats/train_feats.parquet')
test_feats.to_parquet('data_eng/feats/test_feats.parquet')

client.shutdown()