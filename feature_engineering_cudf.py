import cudf as dd
from feature_engineering_2 import (
    pos_cash, process_unified, process_bureau_and_balance, 
    process_previous_applications, installments_payments,
    credit_card_balance
    )

# initiating mem management
# this allows for spilling out of the gpu ram 
dd.set_allocator("managed")

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

train_index = train.index
test_index = test.index

train_target = train['TARGET']
unified = dd.concat([train.drop('TARGET', axis=1), test])
print("starting processing")

unified_feat = process_unified(unified, dd)

bureau_agg = process_bureau_and_balance(bureau, bureau_balance, dd)

############################

prev_agg = process_previous_applications(prev, dd)
pos_agg = pos_cash(pc_balance, dd)
ins_agg = installments_payments(payments, dd)
cc_agg = credit_card_balance(cc_balance, dd)

unified_feat = unified_feat.join(bureau_agg, how='left', on='SK_ID_CURR') \
    .join(prev_agg, how='left', on='SK_ID_CURR') \
    .join(pos_agg, how='left', on='SK_ID_CURR') \
    .join(ins_agg, how='left', on='SK_ID_CURR') \
    .join(cc_agg, how='left', on='SK_ID_CURR')

########################

# basic fallback in case we get way too many columns
#unified_feat = unified_feat.join(bureau_agg, how='left', on='SK_ID_CURR')

# we can't use bool column types in xgb later on
bool_columns = [col for col in unified_feat.columns if (unified_feat[col].dtype in ['bool']) ]    
unified_feat[bool_columns] = unified_feat[bool_columns].astype('int64')

from sklearn.preprocessing import LabelEncoder
# label encode cats
label_encode_dict = {}

categorical = unified_feat.select_dtypes(include=pd.CategoricalDtype).columns 
for column in categorical:
    label_encode_dict[column] = LabelEncoder()
    unified_feat[column] =  label_encode_dict[column].fit_transform(unified_feat[column])
    unified_feat[column] = unified_feat[column].astype('int64')

### Fix for Int64D
Int64D = unified_feat.select_dtypes(include=[pd.Int64Dtype]).columns
unified_feat[Int64D] = unified_feat[Int64D].fillna(0)
unified_feat[Int64D] = unified_feat[Int64D].astype('int64')

### fix unit8
uint8 = unified_feat.select_dtypes(include=['uint8']).columns
unified_feat[uint8] = unified_feat[uint8].astype('int64')

#unified_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
na_cols = unified_feat.isna().any()[unified_feat.isna().any()==True].index.to_arrow().to_pylist()
unified_feat[na_cols] = unified_feat[na_cols].fillna(0)

train_feats = unified_feat.loc[train_index].merge(train_target, how='left', 
                                               left_index=True, right_index=True)
test_feats = unified_feat.loc[test_index]

train_feats.to_parquet('data_eng/feats/train_feats.parquet')
test_feats.to_parquet('data_eng/feats/test_feats.parquet')
