## Feature Engineering using dask

import time

import pandas as pd
from feature_engineering import feature_engineering

### Load Data
bureau_balance = pd.read_csv('data/bureau_balance.csv')
bureau = pd.read_csv('data/bureau.csv')
cc_balance = pd.read_csv('data/credit_card_balance.csv')
payments = pd.read_csv('data/installments_payments.csv')
pc_balance = pd.read_csv('data/POS_CASH_balance.csv')
prev = pd.read_csv('data/previous_application.csv')
train = pd.read_csv('data/application_train.csv')
test = pd.read_csv('data/application_test.csv')

# need to split out the parquet writing
# also need to fix a UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
avg_bureau, sum_cc_balance, sum_payments, sum_pc_balance, sum_prev, train_feat = feature_engineering(bureau_balance, bureau, cc_balance, payments, pc_balance,
                        prev, train, test)

avg_bureau.to_parquet(path='data_eng/pandas/avg_bureau.parq')
sum_cc_balance.to_parquet(path='data_eng/pandas/sum_cc_balance.parq')
sum_payments.to_parquet(path='data_eng/pandas/sum_payments.parq')
sum_pc_balance.to_parquet(path='data_eng/pandas/sum_pc_balance.parq')
sum_prev.to_parquet(path='data_eng/pandas/sum_prev.parq')
train_feat.to_parquet(path='data_eng/pandas/train_feat.parq')
