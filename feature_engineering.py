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
    avg_bbalance['MONTHS_BALANCE_std'] = avg_bbalance.MONTHS_BALANCE_std.fillna(0)

    # free up gpu ram
    del(bureau_balance)
    gc.collect()

    print("procecssing cc balance")

    ## fill nulls for cc_balance
    cc_balance['AMT_DRAWINGS_ATM_CURRENT'] = cc_balance['AMT_DRAWINGS_ATM_CURRENT'].fillna(0)
    cc_balance['AMT_DRAWINGS_OTHER_CURRENT'] = cc_balance['AMT_DRAWINGS_OTHER_CURRENT'].fillna(0)
    cc_balance['AMT_DRAWINGS_POS_CURRENT'] = cc_balance['AMT_DRAWINGS_POS_CURRENT'].fillna(0)
    cc_balance['AMT_INST_MIN_REGULARITY'] = cc_balance['AMT_INST_MIN_REGULARITY'].fillna(0)
    cc_balance['AMT_PAYMENT_CURRENT'] = cc_balance['AMT_PAYMENT_CURRENT'].fillna(0)
    cc_balance['CNT_DRAWINGS_ATM_CURRENT'] = cc_balance['CNT_DRAWINGS_ATM_CURRENT'].fillna(0)
    cc_balance['CNT_DRAWINGS_OTHER_CURRENT'] = cc_balance['CNT_DRAWINGS_OTHER_CURRENT'].fillna(0)
    cc_balance['CNT_DRAWINGS_POS_CURRENT'] = cc_balance['CNT_DRAWINGS_POS_CURRENT'].fillna(0)
    cc_balance['CNT_INSTALMENT_MATURE_CUM'] = cc_balance['CNT_INSTALMENT_MATURE_CUM'].fillna(0)

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
    
    # fill nulls for bureau
    bureau['DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_ENDDATE'].fillna(0)
    bureau['DAYS_ENDDATE_FACT'] = bureau['DAYS_ENDDATE_FACT'].fillna(0)
    bureau['AMT_CREDIT_MAX_OVERDUE'] = bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(0)
    bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    bureau['AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM_LIMIT'].fillna(0)
    bureau['AMT_ANNUITY'] = bureau['AMT_ANNUITY'].fillna(0)

    avg_bureau = bureau.merge(avg_bbalance, how='left', 
                              left_on='SK_ID_BUREAU', 
                              right_index=True)

    avg_bureau.set_index('SK_ID_CURR')

    ## free up gpu ram
    del(bureau)
    del(avg_bbalance)
    gc.collect()

    print("procecssing payments")

    payments['DAYS_ENTRY_PAYMENT'] = payments['DAYS_ENTRY_PAYMENT'].fillna(0)
    payments['AMT_PAYMENT'] = payments['AMT_PAYMENT'].fillna(0)

    ## Buld Payments
    sum_payments = payments.drop('SK_ID_PREV', axis=1)
    sum_payments['PAYMENT_PERC'] = sum_payments.AMT_PAYMENT / sum_payments.AMT_INSTALMENT
    sum_payments['PAYMENT_PERC'] = sum_payments['PAYMENT_PERC'].fillna(0)
    sum_payments['PAYMENT_DIFF'] = sum_payments.AMT_INSTALMENT - sum_payments.AMT_PAYMENT
    sum_payments['DPD'] = sum_payments.DAYS_ENTRY_PAYMENT - sum_payments.DAYS_INSTALMENT
    sum_payments['DBD'] = sum_payments.DAYS_INSTALMENT - sum_payments.DAYS_ENTRY_PAYMENT
    
    # turn negatives into 0
    sum_payments['DPD'] = sum_payments['DPD'].map(lambda x: x if x > 0 else 0)
    sum_payments['DBD'] = sum_payments['DBD'].map(lambda x: x if x > 0 else 0)

    # group and apply our aggs
    sum_payments = sum_payments.select_dtypes('number').groupby('SK_ID_CURR') \
                .agg(agg_func)

    # something weird is happening here re column naming
    sum_payments.columns = ["_".join(x) for x in sum_payments.columns.ravel()]

    # Fill back nulls
    sum_payments['NUM_INSTALMENT_VERSION_std'] = sum_payments['NUM_INSTALMENT_VERSION_std'].fillna(0)
    sum_payments['NUM_INSTALMENT_NUMBER_std'] = sum_payments['NUM_INSTALMENT_NUMBER_std'].fillna(0)
    sum_payments['DAYS_INSTALMENT_std'] = sum_payments['DAYS_INSTALMENT_std'].fillna(0)
    sum_payments['DAYS_ENTRY_PAYMENT_std'] = sum_payments['DAYS_ENTRY_PAYMENT_std'].fillna(0)
    sum_payments['AMT_INSTALMENT_std'] = sum_payments['AMT_INSTALMENT_std'].fillna(0)
    sum_payments['AMT_PAYMENT_std'] = sum_payments['AMT_PAYMENT_std'].fillna(0)
    sum_payments['PAYMENT_PERC_std'] = sum_payments['PAYMENT_PERC_std'].fillna(0)
    sum_payments['PAYMENT_DIFF_std'] = sum_payments['PAYMENT_DIFF_std'].fillna(0)
    sum_payments['DPD_std'] = sum_payments['DPD_std'].fillna(0)
    sum_payments['DBD_std'] = sum_payments['DBD_std'].fillna(0)
    

    #sum_payments.set_index('SK_ID_CURR')

    del(payments)
    gc.collect()

    print("processing pc_balance")

    ## Build Sum_PC_Balance

    ## fill nulls - fill 0 is fine as that means we don't owe other credit
    pc_balance['CNT_INSTALMENT'] = pc_balance['CNT_INSTALMENT'].fillna(0)
    pc_balance['CNT_INSTALMENT'] = pc_balance['CNT_INSTALMENT_FUTURE'].fillna(0)

    sum_pc_balance = pc_balance.drop('SK_ID_PREV', axis=1).select_dtypes('number').groupby('SK_ID_CURR') \
                .agg(agg_func)

    sum_pc_balance.columns = ["_".join(x) for x in sum_pc_balance.columns.ravel()]
    
    sum_pc_balance['CNT_INSTALMENT_std'] = sum_pc_balance['CNT_INSTALMENT_std'].fillna(0)
    sum_pc_balance['CNT_INSTALMENT_FUTURE_mean'] = sum_pc_balance['CNT_INSTALMENT_FUTURE_mean'].fillna(0)
    sum_pc_balance['CNT_INSTALMENT_FUTURE_max'] = sum_pc_balance['CNT_INSTALMENT_FUTURE_max'].fillna(0)
    sum_pc_balance['CNT_INSTALMENT_FUTURE_min'] = sum_pc_balance['CNT_INSTALMENT_FUTURE_min'].fillna(0)
    sum_pc_balance['CNT_INSTALMENT_FUTURE_sum'] = sum_pc_balance['CNT_INSTALMENT_FUTURE_sum'].fillna(0)
    sum_pc_balance['CNT_INSTALMENT_FUTURE_std'] = sum_pc_balance['CNT_INSTALMENT_FUTURE_std'].fillna(0)
    sum_pc_balance['SK_DPD_std'] = sum_pc_balance['SK_DPD_std'].fillna(0)
    sum_pc_balance['SK_DPD_DEF_std'] = sum_pc_balance['SK_DPD_DEF_std'].fillna(0)
    sum_pc_balance['MONTHS_BALANCE_std'] = sum_pc_balance['MONTHS_BALANCE_std'].fillna(0)

    # free up gpu ram
    del(pc_balance)
    gc.collect()

    print("processing prev table")

    ## Build Sum_Prev
    prev = prev.drop('SK_ID_PREV', axis=1)
    prev['NFLAG_INSURED_ON_APPROVAL'] = prev['NFLAG_INSURED_ON_APPROVAL'].astype('object')
    prev['NFLAG_INSURED_ON_APPROVAL'] = prev['NFLAG_INSURED_ON_APPROVAL'].fillna('None')
    prev['NFLAG_INSURED_ON_APPROVAL'] = prev['NFLAG_INSURED_ON_APPROVAL'].astype('category')

    prev['NAME_TYPE_SUITE'].fillna('Unknown', inplace=True)
    prev['NAME_TYPE_SUITE'] = prev['NAME_TYPE_SUITE'].astype('category')    

    prev['PRODUCT_COMBINATION'].fillna('None', inplace=True)
    prev['PRODUCT_COMBINATION'] = prev['PRODUCT_COMBINATION'].astype('category')

    prev['AMT_ANNUITY'].fillna(0, inplace=True)
    prev['AMT_CREDIT'].fillna(0, inplace=True)
    prev['AMT_DOWN_PAYMENT'].fillna(0, inplace=True)
    prev['AMT_GOODS_PRICE'].fillna(0, inplace=True)
    prev['RATE_DOWN_PAYMENT'].fillna(0, inplace=True)
    prev['RATE_INTEREST_PRIMARY'].fillna(0, inplace=True)
    prev['RATE_INTEREST_PRIVILEGED'].fillna(0, inplace=True)
    prev['CNT_PAYMENT'].fillna(0, inplace=True)
    prev['DAYS_FIRST_DRAWING'].fillna(0, inplace=True)
    prev['DAYS_FIRST_DUE'].fillna(0, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].fillna(0, inplace=True)
    prev['DAYS_LAST_DUE'].fillna(0, inplace=True)
    prev['DAYS_TERMINATION'].fillna(0, inplace=True)

    prev['DAYS_FIRST_DRAWING'] = prev.DAYS_FIRST_DRAWING.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_FIRST_DUE'] = prev.DAYS_FIRST_DUE.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_LAST_DUE_1ST_VERSION'] = prev.DAYS_LAST_DUE_1ST_VERSION.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_LAST_DUE'] = prev.DAYS_LAST_DUE.map(lambda x: np.nan if x == 365243 else x)
    prev['DAYS_TERMINATION'] = prev.DAYS_TERMINATION.map(lambda x: np.nan if x == 365243 else x)
    prev['APP_CREDIT_PERC'] = prev.AMT_APPLICATION / prev.AMT_CREDIT

    sum_prev = prev.select_dtypes('number').groupby('SK_ID_CURR') \
                .agg(agg_func)

    sum_prev.columns = ["_".join(x) for x in sum_prev.columns.ravel()]

    #sum_prev['SK_ID_PREV_std'].fillna(0, inplace=True)
    
    # these are all the std columns so all good
    # to arrow is only needed for cudf .to_arrow()
    for column in sum_prev.isnull().any()[sum_prev.isnull().any()==True].index.tolist():
        sum_prev[column].fillna(0, inplace=True)

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
