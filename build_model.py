## build Model dask_cudf


import time
from dask.distributed import Client, wait
import numpy as np
import dask.dataframe as dd

from dask_cuda import LocalCUDACluster
import dask_cudf
import cudf

#cluster = LocalCUDACluster()
#client = Client(cluster)
#client

client = Client(n_workers=2, threads_per_worker=8, processes=False, memory_limit='20GB')
client

## reload our parquet
avg_bureau = dd.read_parquet('data_eng/avg_bureau.parq')
sum_cc_balance = dd.read_parquet(path='data_eng/sum_cc_balance.parq')
sum_payments = dd.read_parquet(path='data_eng/sum_payments.parq')
sum_pc_balance = dd.read_parquet(path='data_eng/sum_pc_balance.parq')
sum_prev = dd.read_parquet(path='data_eng/sum_prev.parq')
train = dd.read_csv('data/application_train.csv')

## Lets try and merge everything with dask this time
train_feat = train.drop('TARGET', axis=1) \
    .merge(avg_bureau, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR') \
    .merge(sum_cc_balance, how='left', left_on='SK_ID_CURR', right_index=True) \
    .merge(sum_payments, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR') \
    .merge(sum_pc_balance, how='left', left_on='SK_ID_CURR', right_index=True) \
    .merge(sum_prev, how='left', left_on='SK_ID_CURR', right_index=True) \
    .drop('SK_ID_CURR', axis=1)

train_feat['DAYS_EMPLOYED'] = train_feat.DAYS_EMPLOYED.map(lambda x: np.nan if x == 365243 else x)
train_feat['DAYS_EMPLOYED_PERC'] = np.sqrt(train_feat.DAYS_EMPLOYED / train_feat.DAYS_BIRTH)
train_feat['INCOME_CREDIT_PERC'] = train_feat.AMT_INCOME_TOTAL / train_feat.AMT_CREDIT
train_feat['INCOME_PER_PERSON'] = np.log1p(train_feat.AMT_INCOME_TOTAL / train_feat.CNT_FAM_MEMBERS)

train_feat['ANNUITY_INCOME_PERC'] = np.sqrt(train_feat.AMT_ANNUITY / (1 + train_feat.AMT_INCOME_TOTAL)) 
train_feat['LOAN_INCOME_RATIO'] = train_feat.AMT_CREDIT / train_feat.AMT_INCOME_TOTAL
train_feat['ANNUITY_LENGTH'] =  train_feat.AMT_CREDIT / train_feat.AMT_ANNUITY
train_feat['CHILDREN_RATIO'] = train_feat.CNT_CHILDREN / train_feat.CNT_FAM_MEMBERS 
train_feat['CREDIT_TO_GOODS_RATIO'] = train_feat.AMT_CREDIT / train_feat.AMT_GOODS_PRICE 
train_feat['INC_PER_CHLD'] = train_feat.AMT_INCOME_TOTAL / (1 + train_feat.CNT_CHILDREN)
train_feat['SOURCES_PROD'] = train_feat.EXT_SOURCE_1 * train_feat.EXT_SOURCE_2 * train_feat.EXT_SOURCE_3
train_feat['CAR_TO_BIRTH_RATIO'] = train_feat.OWN_CAR_AGE / train_feat.DAYS_BIRTH
train_feat['CAR_TO_EMPLOY_RATIO'] = train_feat.OWN_CAR_AGE / train_feat.DAYS_EMPLOYED
train_feat['PHONE_TO_BIRTH_RATIO'] = train_feat.DAYS_LAST_PHONE_CHANGE / train_feat.DAYS_BIRTH
train_feat['PHONE_TO_EMPLOY_RATIO'] = train_feat.DAYS_LAST_PHONE_CHANGE / train_feat.DAYS_EMPLOYED





train_feat.to_parquet(path='data_eng/train_feat')
