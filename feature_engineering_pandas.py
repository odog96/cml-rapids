## Feature Engineering using dask

import time

import pandas as dd
from feature_engineering_2 import process_unified, process_bureau_and_balance, process_previous_applications

### Load Data
bureau_balance = dd.read_parquet('raw_data/bureau_balance.parquet')
bureau = dd.read_parquet('raw_data/bureau.parquet')

# behaviour data linked to prev as well as current loan
cc_balance = dd.read_parquet('raw_data/cc_balance.parquet')
payments = dd.read_parquet('raw_data/payments.parquet')
pc_balance = dd.read_parquet('raw_data/pc_balance.parquet')

prev = dd.read_parquet('raw_data/prev.parquet')

train = dd.read_parquet('raw_data/train.parquet')
test = dd.read_parquet('raw_data/test.parquet')

train_index = train.index
test_index = test.index

train_target = train['TARGET']
unified = dd.concat([train.drop('TARGET', axis=1), test])

# need to split out the parquet writing
# also need to fix a UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access

unified_feat = process_unified(unified, dd)

bureau_agg = process_bureau_and_balance(bureau, bureau_balance, dd)

unified_feat = unified_feat.join(bureau_agg, how='left', on='SK_ID_CURR')

# we can't use bool column types in xgb later on
bool_columns = [col for col in unified_feat.columns if (unified_feat[col].dtype in ['bool']) ]    
unified_feat[bool_columns] = unified_feat[bool_columns].astype('int64')

train_feats = unified_feat.loc[train_index].merge(train_target, how='left', 
                                               left_index=True, right_index=True)
test_feats = unified_feat.loc[test_index]

train_feats.to_parquet('data_eng/feats/train_feats.parquet')
test_feats.to_parquet('data_eng/feats/test_feats.parquet')