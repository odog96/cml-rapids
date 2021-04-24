import cudf as dd
from feature_engineering_adv import feature_engineering

### Load datasets
print("loading data")
bureau_balance = dd.read_parquet('raw_data/bureau_balance.parquet')
bureau = dd.read_parquet('raw_data/bureau.parquet')
cc_balance = dd.read_parquet('raw_data/cc_balance.parquet')
payments = dd.read_parquet('raw_data/payments.parquet')
pc_balance = dd.read_parquet('raw_data/pc_balance.parquet')
prev = dd.read_parquet('raw_data/prev.parquet')
train = dd.read_parquet('raw_data/train.parquet')
test = dd.read_parquet('raw_data/test.parquet')
train_target = train['TARGET']
unified = dd.concat([train.drop('TARGET', axis=1), test])
print("starting processing")

unified_feat = feature_engineering(bureau_balance, bureau, 
                                    cc_balance, payments, pc_balance,
                                    prev, unified, dd)

train_rows = train.shape
train_feats = unified_feat.iloc[:307511].merge(train_target, how='left', 
                                           left_index=True, right_index=True)
test_feats = unified_feat.iloc[307511:]
#avg_bureau.to_parquet(path='data_eng/avg_bureau')
#sum_cc_balance.to_parquet(path='data_eng/sum_cc_balance')
#sum_payments.to_parquet(path='data_eng/sum_payments')
#sum_pc_balance.to_parquet(path='data_eng/sum_pc_balance')
#sum_prev.to_parquet(path='data_eng/sum_prev')

train_feats.to_parquet('data_eng/feats/train_feats.parquet')
test_feats.to_parquet('data_eng/feats/test_feats.parquet')
