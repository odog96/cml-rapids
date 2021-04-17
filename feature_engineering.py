### Data Engineering Job

### Set conda env?
#%conda activate rapids-0.18

import cudf
import numpy as np

cudf.set_allocator("managed")

### Load datasets
bureau_balance = cudf.read_csv('data/bureau_balance.csv')
bureau = cudf.read_csv('data/bureau.csv')
cc_balance = cudf.read_csv('data/credit_card_balance.csv')
payments = cudf.read_csv('data/installments_payments.csv')
pc_balance = cudf.read_csv('data/POS_CASH_balance.csv')
prev = cudf.read_csv('data/previous_application.csv')
train = cudf.read_csv('data/application_train.csv')
test = cudf.read_csv('data/application_test.csv')

## aggregation functions for our groupings
agg_func = ['mean', 'max', 'min', 'sum', 'std']

## Build Average Bureau Balance
avg_bbalance = bureau_balance.select_dtypes('number') \
                .groupby('SK_ID_BUREAU').agg(agg_func)

avg_bbalance.columns = ["_".join(x) for x in avg_bbalance.columns.ravel()]

## Build sum Credit Card Balance
sum_cc_balance = cc_balance.drop('SK_ID_PREV', axis=1) \
                    .select_dtypes('number').groupby('SK_ID_CURR') \
                    .agg(agg_func)

sum_cc_balance.columns = ["_".join(x) for x in sum_cc_balance.columns.ravel()]

## Build Avg Bureau table
avg_bureau = bureau.merge(avg_bbalance, how='left', 
                          left_on='SK_ID_BUREAU', 
                          right_index=True)

## Buld Payments
sum_payments = payments.drop('SK_ID_PREV', axis=1)
sum_payments['PAYMENT_PERC'] = sum_payments.AMT_PAYMENT / sum_payments.AMT_INSTALMENT
sum_payments['PAYMENT_DIFF'] = sum_payments.AMT_INSTALMENT - sum_payments.AMT_PAYMENT
sum_payments['DPD'] = sum_payments.DAYS_ENTRY_PAYMENT - sum_payments.DAYS_INSTALMENT
sum_payments['DBD'] = sum_payments.DAYS_INSTALMENT - sum_payments.DAYS_ENTRY_PAYMENT
sum_payments['DPD'] = sum_payments['DPD']
sum_payments['DBD'] = sum_payments['DBD']

## Build Sum_PC_Balance
sum_pc_balance = pc_balance.drop('SK_ID_PREV', axis=1).select_dtypes('number').groupby('SK_ID_CURR') \
            .agg(agg_func)

sum_pc_balance.columns = ["_".join(x) for x in sum_pc_balance.columns.ravel()]

## Build Sum_Prev
prev = prev.drop('SK_ID_PREV', axis=1)
prev.DAYS_FIRST_DRAWING = prev.DAYS_FIRST_DRAWING.map(lambda x: np.nan if x == 365243 else x)
prev.DAYS_FIRST_DUE = prev.DAYS_FIRST_DUE.map(lambda x: np.nan if x == 365243 else x)
prev.DAYS_LAST_DUE_1ST_VERSION = prev.DAYS_LAST_DUE_1ST_VERSION.map(lambda x: np.nan if x == 365243 else x)
prev.DAYS_LAST_DUE = prev.DAYS_LAST_DUE.map(lambda x: np.nan if x == 365243 else x)
prev.DAYS_TERMINATION = prev.DAYS_TERMINATION.map(lambda x: np.nan if x == 365243 else x)
prev.APP_CREDIT_PERC = prev.AMT_APPLICATION / prev.AMT_CREDIT

sum_prev = prev.select_dtypes('number').groupby('SK_ID_CURR') \
            .agg(agg_func)

sum_prev.columns = ["_".join(x) for x in sum_prev.columns.ravel()]

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


## Outputs - to write out
#avg_bureau.to_pandas().to_parquet(path='data_eng/avg_bureau.parq')
#sum_cc_balance.to_pandas().to_parquet(path='data_eng/sum_cc_balance.parq')
#sum_payments.to_pandas().to_parquet(path='data_eng/sum_payments.parq')
#sum_pc_balance.to_pandas().to_parquet(path='data_eng/sum_pc_balance.parq')
#sum_prev.to_pandas().to_parquet(path='data_eng/sum_prev.parq')