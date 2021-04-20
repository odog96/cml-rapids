### Data Engineering Job

#import numpy as np
import cudf as dd

## activate auto memory management
## allows for spill
#cudf.set_allocator("managed")


def feature_engineering(bureau_balance, bureau, cc_balance, payments, pc_balance,
                        prev, unified):
    """

    Feature engineering script to process our data
    we split this into it's own function so that we can use it across 
    different functions easier

    """
    import numpy as np
    import gc

    ## aggregation functions for our groupings
    agg_func = ['mean', 'max', 'min', 'sum', 'std']

    print("procecssing bureau balance")

    ## Build Average Bureau Balance
    avg_bbalance = bureau_balance.select_dtypes('number') \
                    .groupby('SK_ID_BUREAU').agg(agg_func)

    avg_bbalance.columns = ["_".join(x) for x in avg_bbalance.columns.ravel()]

    # free up gpu ram
    del(bureau_balance)
    gc.collect()

    print("procecssing cc balance")

    ## Build sum Credit Card Balance
    sum_cc_balance = cc_balance.drop('SK_ID_PREV', axis=1) \
                        .select_dtypes('number').groupby('SK_ID_CURR') \
                        .agg(agg_func)

    sum_cc_balance.columns = ["_".join(x) for x in sum_cc_balance.columns.ravel()]

    # free up gpu ram
    del(cc_balance)
    gc.collect()

    print("procecssing bureau")

    ## Build Avg Bureau table
    avg_bureau = bureau.merge(avg_bbalance, how='left', 
                              left_on='SK_ID_BUREAU', 
                              right_index=True)

    avg_bureau.set_index('SK_ID_CURR')

    ## free up gpu ram
    del(bureau)
    del(avg_bbalance)
    gc.collect()

    print("procecssing payments")

    ## Buld Payments
    sum_payments = payments.drop('SK_ID_PREV', axis=1)
    sum_payments['PAYMENT_PERC'] = sum_payments.AMT_PAYMENT / sum_payments.AMT_INSTALMENT
    sum_payments['PAYMENT_DIFF'] = sum_payments.AMT_INSTALMENT - sum_payments.AMT_PAYMENT
    sum_payments['DPD'] = sum_payments.DAYS_ENTRY_PAYMENT - sum_payments.DAYS_INSTALMENT
    sum_payments['DBD'] = sum_payments.DAYS_INSTALMENT - sum_payments.DAYS_ENTRY_PAYMENT
    sum_payments['DPD'] = sum_payments['DPD']
    sum_payments['DBD'] = sum_payments['DBD']

    sum_payments.set_index('SK_ID_CURR')

    del(payments)
    gc.collect()

    print("processing pc_balance")

    ## Build Sum_PC_Balance
    sum_pc_balance = pc_balance.drop('SK_ID_PREV', axis=1).select_dtypes('number').groupby('SK_ID_CURR') \
                .agg(agg_func)

    sum_pc_balance.columns = ["_".join(x) for x in sum_pc_balance.columns.ravel()]

    # free up gpu ram
    del(pc_balance)
    gc.collect()

    print("processing prev table")

    ## Build Sum_Prev
    prev = prev.drop('SK_ID_PREV', axis=1)
    prev['DAYS_FIRST_DRAWING'] = prev.DAYS_FIRST_DRAWING.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_FIRST_DUE'] = prev.DAYS_FIRST_DUE.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_LAST_DUE_1ST_VERSION'] = prev.DAYS_LAST_DUE_1ST_VERSION.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_LAST_DUE'] = prev.DAYS_LAST_DUE.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_TERMINATION'] = prev.DAYS_TERMINATION.map(lambda x: np.nan if x == 365243 else x)
    prev['APP_CREDIT_PERC'] = prev.AMT_APPLICATION / prev.AMT_CREDIT

    sum_prev = prev.select_dtypes('number').groupby('SK_ID_CURR') \
                .agg(agg_func)

    sum_prev.columns = ["_".join(x) for x in sum_prev.columns.ravel()]

    # dask test - WIP
    #sum_prev.compute()

    # free up gpu ram
    del(prev)
    gc.collect()

    print("merging feats into train and test - part1")

    #train.set_index('SK_ID_CURR')

    feats = unified \
        .merge(avg_bureau, how='left', left_index=True, right_index=True) \
        .merge(sum_cc_balance, how='left', left_index=True, right_index=True) \
        .merge(sum_payments, how='left', left_index=True, right_index=True) \
        .merge(sum_pc_balance, how='left', left_index=True, right_index=True) \
        .merge(sum_prev, how='left', left_index=True, right_index=True) \
        #.drop('SK_ID_CURR', axis=1)

    #del(sum_pc_balance)
    #del(sum_prev)
    #gc.collect()

    print("extra feats")

    feats['DAYS_EMPLOYED'] = feats.DAYS_EMPLOYED.map(lambda x: np.nan if x == 365243 else x)
    feats['DAYS_EMPLOYED_PERC'] = np.sqrt(feats.DAYS_EMPLOYED / feats.DAYS_BIRTH)
    feats['INCOME_CREDIT_PERC'] = feats.AMT_INCOME_TOTAL / feats.AMT_CREDIT
    
    # need to debug this
    #train_feat['INCOME_PER_PERSON'] = np.log1p(train_feat.AMT_INCOME_TOTAL / train_feat.CNT_FAM_MEMBERS)

    print("feats done")

    #train_feat = feats[]

    # dask test - WIP
    #train_feat.compute()

    return avg_bureau, sum_cc_balance, sum_payments, sum_pc_balance, sum_prev, feats


if __name__ == '__main__':

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

    avg_bureau, sum_cc_balance, sum_payments, \
        sum_pc_balance, sum_prev, unified_feat = feature_engineering(bureau_balance, bureau, 
                                                    cc_balance, payments, pc_balance,
                                                    prev, unified)

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
