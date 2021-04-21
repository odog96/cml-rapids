### Data Engineering Job

import cudf as dd
import numpy as np
import gc

def process_bureau_balance(bureau_balance, agg_func):
    """

    this script processes the bureau balance table and produces a set of feature for merging back with the main training table
    Args:
        bureau_balance (dataframe): The Bureau Balance table as a DataFrame cudf or pandas is okay
        agg_func (list): List of functions for aggregating the numeric values into one line item per SK_ID_BUREAU 

    Returns:
        avg_bbalance (dataframe): The processed dataframe for the next step

    """

    avg_bbalance = bureau_balance.select_dtypes('number') \
                    .groupby('SK_ID_BUREAU').agg(agg_func)

    avg_bbalance.columns = ["_".join(x) for x in avg_bbalance.columns.ravel()]
    avg_bbalance['MONTHS_BALANCE_std'] = avg_bbalance.MONTHS_BALANCE_std.fillna(0)

    # free up gpu ram
    del(bureau_balance)
    gc.collect()

    return avg_bbalance

def process_cc_balance(cc_balance, agg_func):
    """

    this script processes the credit card balance table and produces a set of feature for merging back with the main training table
    Args:
        cc_balance (dataframe): The credit card Balance table as a DataFrame cudf or pandas is okay
        agg_func (list): List of functions for aggregating the numeric values into one line item per SK_ID_BUREAU 

    Returns:
        sum_cc_balance (dataframe): The processed dataframe for the next step

    """
    
    ## fill nulls for cc_balance
    cc_balance['AMT_DRAWINGS_ATM_CURRENT'].fillna(0, inplace=True)
    cc_balance['AMT_DRAWINGS_OTHER_CURRENT'].fillna(0, inplace=True)
    cc_balance['AMT_DRAWINGS_POS_CURRENT'].fillna(0, inplace=True)
    cc_balance['AMT_INST_MIN_REGULARITY'].fillna(0, inplace=True)
    cc_balance['AMT_PAYMENT_CURRENT'].fillna(0, inplace=True)
    cc_balance['CNT_DRAWINGS_ATM_CURRENT'].fillna(0, inplace=True)
    cc_balance['CNT_DRAWINGS_OTHER_CURRENT'].fillna(0, inplace=True)
    cc_balance['CNT_DRAWINGS_POS_CURRENT'].fillna(0, inplace=True)
    cc_balance['CNT_INSTALMENT_MATURE_CUM'].fillna(0, inplace=True)

    ## Build sum Credit Card Balance
    sum_cc_balance = cc_balance.drop('SK_ID_PREV', axis=1) \
                        .select_dtypes('number').groupby('SK_ID_CURR') \
                        .agg(agg_func)

    sum_cc_balance.columns = ["_".join(x) for x in sum_cc_balance.columns.ravel()]

    cols_to_fill = ['MONTHS_BALANCE_mean', 'MONTHS_BALANCE_max', 'MONTHS_BALANCE_min',
        'MONTHS_BALANCE_sum', 'MONTHS_BALANCE_std', 'AMT_BALANCE_mean',
        'AMT_BALANCE_max', 'AMT_BALANCE_min', 'AMT_BALANCE_sum', 'AMT_BALANCE_std',
        'AMT_CREDIT_LIMIT_ACTUAL_mean', 'AMT_CREDIT_LIMIT_ACTUAL_max', 'AMT_CREDIT_LIMIT_ACTUAL_min',
        'AMT_CREDIT_LIMIT_ACTUAL_sum', 'AMT_CREDIT_LIMIT_ACTUAL_std', 'AMT_DRAWINGS_ATM_CURRENT_mean',
        'AMT_DRAWINGS_ATM_CURRENT_max', 'AMT_DRAWINGS_ATM_CURRENT_min', 'AMT_DRAWINGS_ATM_CURRENT_sum',
        'AMT_DRAWINGS_ATM_CURRENT_std', 'AMT_DRAWINGS_CURRENT_mean', 'AMT_DRAWINGS_CURRENT_max',
        'AMT_DRAWINGS_CURRENT_min', 'AMT_DRAWINGS_CURRENT_sum', 'AMT_DRAWINGS_CURRENT_std',
        'AMT_DRAWINGS_OTHER_CURRENT_mean', 'AMT_DRAWINGS_OTHER_CURRENT_max', 'AMT_DRAWINGS_OTHER_CURRENT_min',
        'AMT_DRAWINGS_OTHER_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_std', 'AMT_DRAWINGS_POS_CURRENT_mean',
        'AMT_DRAWINGS_POS_CURRENT_max', 'AMT_DRAWINGS_POS_CURRENT_min', 'AMT_DRAWINGS_POS_CURRENT_sum',
        'AMT_DRAWINGS_POS_CURRENT_std', 'AMT_INST_MIN_REGULARITY_mean', 'AMT_INST_MIN_REGULARITY_max',
        'AMT_INST_MIN_REGULARITY_min', 'AMT_INST_MIN_REGULARITY_sum', 'AMT_INST_MIN_REGULARITY_std',
        'AMT_PAYMENT_CURRENT_mean', 'AMT_PAYMENT_CURRENT_max', 'AMT_PAYMENT_CURRENT_min',
        'AMT_PAYMENT_CURRENT_sum', 'AMT_PAYMENT_CURRENT_std', 'AMT_PAYMENT_TOTAL_CURRENT_mean',
        'AMT_PAYMENT_TOTAL_CURRENT_max', 'AMT_PAYMENT_TOTAL_CURRENT_min', 'AMT_PAYMENT_TOTAL_CURRENT_sum',
        'AMT_PAYMENT_TOTAL_CURRENT_std', 'AMT_RECEIVABLE_PRINCIPAL_mean', 'AMT_RECEIVABLE_PRINCIPAL_max',
        'AMT_RECEIVABLE_PRINCIPAL_min', 'AMT_RECEIVABLE_PRINCIPAL_sum', 'AMT_RECEIVABLE_PRINCIPAL_std',
        'AMT_RECIVABLE_mean', 'AMT_RECIVABLE_max', 'AMT_RECIVABLE_min', 'AMT_RECIVABLE_sum',
        'AMT_RECIVABLE_std', 'AMT_TOTAL_RECEIVABLE_mean', 'AMT_TOTAL_RECEIVABLE_max',
        'AMT_TOTAL_RECEIVABLE_min', 'AMT_TOTAL_RECEIVABLE_sum', 'AMT_TOTAL_RECEIVABLE_std',
        'CNT_DRAWINGS_ATM_CURRENT_mean', 'CNT_DRAWINGS_ATM_CURRENT_max', 'CNT_DRAWINGS_ATM_CURRENT_min',
        'CNT_DRAWINGS_ATM_CURRENT_sum','CNT_DRAWINGS_ATM_CURRENT_std','CNT_DRAWINGS_CURRENT_mean',
        'CNT_DRAWINGS_CURRENT_max','CNT_DRAWINGS_CURRENT_min','CNT_DRAWINGS_CURRENT_sum',
        'CNT_DRAWINGS_CURRENT_std','CNT_DRAWINGS_OTHER_CURRENT_mean','CNT_DRAWINGS_OTHER_CURRENT_max',
        'CNT_DRAWINGS_OTHER_CURRENT_min','CNT_DRAWINGS_OTHER_CURRENT_sum','CNT_DRAWINGS_OTHER_CURRENT_std',
        'CNT_DRAWINGS_POS_CURRENT_mean','CNT_DRAWINGS_POS_CURRENT_max','CNT_DRAWINGS_POS_CURRENT_min',
        'CNT_DRAWINGS_POS_CURRENT_sum','CNT_DRAWINGS_POS_CURRENT_std','CNT_INSTALMENT_MATURE_CUM_mean',
        'CNT_INSTALMENT_MATURE_CUM_max','CNT_INSTALMENT_MATURE_CUM_min','CNT_INSTALMENT_MATURE_CUM_sum',
        'CNT_INSTALMENT_MATURE_CUM_std','SK_DPD_mean','SK_DPD_max','SK_DPD_min','SK_DPD_sum',
        'SK_DPD_std','SK_DPD_DEF_mean','SK_DPD_DEF_max','SK_DPD_DEF_min','SK_DPD_DEF_sum',
        'SK_DPD_DEF_std']
        
    for col in cols_to_fill:
        sum_cc_balance[col] = sum_cc_balance[col].fillna(0)

    # free up gpu ram
    del(cc_balance)
    gc.collect()

    return sum_cc_balance

def process_prev(prev, agg_func):
    """

    this script processes the Previous Loan table and produces a set of feature for merging back with the main training table
    Args:
        prev (dataframe): The credit card Balance table as a DataFrame cudf or pandas is okay
        agg_func (list): List of functions for aggregating the numeric values into one line item per SK_ID_BUREAU 

    Returns:
        sum_prev (dataframe): The processed dataframe for the next step

    """

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

    # these are all the std columns so all good
    sum_prev['AMT_ANNUITY_std'].fillna(0, inplace=True)
    sum_prev['AMT_APPLICATION_std'].fillna(0, inplace=True)
    sum_prev['AMT_CREDIT_std'].fillna(0, inplace=True)
    sum_prev['AMT_DOWN_PAYMENT_std'].fillna(0, inplace=True)
    sum_prev['AMT_GOODS_PRICE_std'].fillna(0, inplace=True)
    sum_prev['HOUR_APPR_PROCESS_START_std'].fillna(0, inplace=True)
    sum_prev['NFLAG_LAST_APPL_IN_DAY_std'].fillna(0, inplace=True)
    sum_prev['RATE_DOWN_PAYMENT_std'].fillna(0, inplace=True)
    sum_prev['RATE_INTEREST_PRIMARY_std'].fillna(0, inplace=True)
    sum_prev['RATE_INTEREST_PRIVILEGED_std'].fillna(0, inplace=True)
    sum_prev['DAYS_DECISION_std'].fillna(0, inplace=True)
    sum_prev['SELLERPLACE_AREA_std'].fillna(0, inplace=True)
    sum_prev['CNT_PAYMENT_std'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DRAWING_max'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DRAWING_min'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DRAWING_mean'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DRAWING_sum'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DRAWING_std'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DUE_max'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DUE_min'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DUE_mean'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DUE_sum'].fillna(0, inplace=True)
    sum_prev['DAYS_FIRST_DUE_std'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_1ST_VERSION_max'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_1ST_VERSION_min'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_1ST_VERSION_mean'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_1ST_VERSION_sum'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_1ST_VERSION_std'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_max'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_min'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_mean'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_sum'].fillna(0, inplace=True)
    sum_prev['DAYS_LAST_DUE_std'].fillna(0, inplace=True)
    sum_prev['DAYS_TERMINATION_max'].fillna(0, inplace=True)
    sum_prev['DAYS_TERMINATION_min'].fillna(0, inplace=True)
    sum_prev['DAYS_TERMINATION_mean'].fillna(0, inplace=True)
    sum_prev['DAYS_TERMINATION_sum'].fillna(0, inplace=True)
    sum_prev['DAYS_TERMINATION_std'].fillna(0, inplace=True)
    sum_prev['APP_CREDIT_PERC_max'].fillna(0, inplace=True)
    sum_prev['APP_CREDIT_PERC_min'].fillna(0, inplace=True)
    sum_prev['APP_CREDIT_PERC_mean'].fillna(0, inplace=True)
    sum_prev['APP_CREDIT_PERC_sum'].fillna(0, inplace=True)
    sum_prev['APP_CREDIT_PERC_std'].fillna(0, inplace=True)

    # free up gpu ram
    del(prev)
    gc.collect()

    return sum_prev

def process_unified(unified):
    """

    this script processes the Unified table and produces a set of feature for merging back with the main training table
    Args:
        unified (dataframe): The unified train and test table as a DataFrame cudf or pandas is okay
        
    Returns:
        unified (dataframe): The processed dataframe for the next step

    """

    bulk_fill = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
        'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 
        'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
        'TOTALAREA_MODE']

    for col in bulk_fill:
        unified[col] = unified[col].fillna(0)

    bulk_fill_2 = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
                    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

    for col in bulk_fill_2:
        unified[col] = unified[col].fillna(int(np.mean(unified[col])))

    bulk_fill_3 = ['DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

    for col in bulk_fill_3:
        unified[col] = unified[col].fillna(0)

    #unified['NAME_TYPE_SUITE'].cat.add_categories('unknown')
    unified['NAME_TYPE_SUITE'] = unified['NAME_TYPE_SUITE'].fillna('unknown')
    unified['NAME_TYPE_SUITE'] = unified['NAME_TYPE_SUITE'].astype('category')
    unified['OWN_CAR_AGE'] = unified['OWN_CAR_AGE'].fillna(np.median(unified['OWN_CAR_AGE']))
    
    unified['OCCUPATION_TYPE'] = unified['OCCUPATION_TYPE'].fillna('unknown')
    unified['OCCUPATION_TYPE'] = unified['OCCUPATION_TYPE'].astype('category')
    unified['CNT_FAM_MEMBERS'] = unified['CNT_FAM_MEMBERS'].fillna(int(np.mean(unified['CNT_FAM_MEMBERS']))) 

    unified['FONDKAPREMONT_MODE'] = unified['FONDKAPREMONT_MODE'].fillna('unknown')
    unified['FONDKAPREMONT_MODE'] = unified['FONDKAPREMONT_MODE'].astype('category')

    unified['HOUSETYPE_MODE'] = unified['HOUSETYPE_MODE'].fillna('unknown')
    unified['HOUSETYPE_MODE'] = unified['HOUSETYPE_MODE'].astype('category')

    unified['WALLSMATERIAL_MODE'] = unified['WALLSMATERIAL_MODE'].fillna('unknown')
    unified['WALLSMATERIAL_MODE'] = unified['WALLSMATERIAL_MODE'].astype('category')

    unified['EMERGENCYSTATE_MODE'] = unified['EMERGENCYSTATE_MODE'].fillna('unknown')
    unified['EMERGENCYSTATE_MODE'] = unified['EMERGENCYSTATE_MODE'].astype('category')

    return unified

def process_bureau(bureau, avg_bbalance):
    """

    this script processes the bureau table and merges it with the avg_bbalance table
    Args:
        bureau (dataframe): the raw bureau table
        avg_bbalance (dataframe): the munged bureau balance table

    Returns:
        avg_bureau (dataframe): The processed Avg bureau Table

    """

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

    avg_bureau['MONTHS_BALANCE_mean'].fillna(0, inplace=True)
    avg_bureau['MONTHS_BALANCE_max'].fillna(0, inplace=True)
    avg_bureau['MONTHS_BALANCE_min'].fillna(0, inplace=True)
    avg_bureau['MONTHS_BALANCE_sum'].fillna(0, inplace=True)
    avg_bureau['MONTHS_BALANCE_std'].fillna(0, inplace=True)

    ## free up gpu ram
    del(bureau)
    del(avg_bbalance)
    gc.collect()

    return avg_bureau

def process_payments(payments, agg_func):
    """

    this script processes the Payments table and produces a set of feature for merging back with the main training table
    Args:
        payments (dataframe): The Payments table as a DataFrame cudf or pandas is okay
        agg_func (list): List of functions for aggregating the numeric values into one line item per SK_ID_BUREAU 

    Returns:
        sum_payments (dataframe): The processed dataframe for the next step

    """

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

    return sum_payments

def process_pc_balance(pc_balance, agg_func):
    """

    this script processes the POS Cash table and produces a set of feature for merging back with the main training table
    Args:
        payments (dataframe): The POS Cash table as a DataFrame cudf or pandas is okay
        agg_func (list): List of functions for aggregating the numeric values into one line item per SK_ID_BUREAU 

    Returns:
        sum_pc_balance (dataframe): The processed dataframe for the next step

    """

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

    return sum_pc_balance

def feature_engineering(bureau_balance, bureau, cc_balance, payments, pc_balance,
                        prev, unified):
    """

    Feature engineering script to process our data
    we split this into it's own function so that we can use it across 
    different backends easier. The transformations for the respective tables are also in their own functions for testing purposes

    Args:
        bureau_balance (dataframe): The raw bureau balance table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)
        bureau (dataframe): The raw bureau table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)
        cc_balance (dataframe): The raw credit card balance table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)
        payments (dataframe): The raw payments table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)
        pc_balance (dataframe): The raw POS Cash balance table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)
        prev (dataframe): The raw previous loan table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)
        unified (dataframe): The raw unified train / test table loaded into a dataframe Pandas/CuDF / Dask_cudf(pending)

    Returns
        avg_bureau (dataframe): The processed avg_bureau table as a dataframe object
        sum_cc_balance (dataframe): The processed sum_cc_balance table as a dataframe object
        sum_payments (dataframe): The processed sum_payments table as a dataframe object
        sum_pc_balance (dataframe): The processed sum_pc_balance table as a dataframe object
        sum_prev (dataframe): The processed sum_prev table as a dataframe object
        feats (dataframe): The processed feats table as a dataframe object

    """

    ## aggregation functions for our groupings
    agg_func = ['mean', 'max', 'min', 'sum', 'std']

    print("procecssing bureau balance")

    avg_bbalance = process_bureau_balance(bureau_balance, agg_func)

    print("procecssing cc balance")

    sum_cc_balance = process_cc_balance(cc_balance, agg_func)

    print("procecssing bureau")

    ## Build Avg Bureau table
    avg_bureau = process_bureau(bureau, avg_bbalance)

    print("procecssing payments")

    sum_payments = process_payments(payments, agg_func)

    print("processing pc_balance")

    ## Build Sum_PC_Balance

    sum_pc_balance = process_pc_balance(pc_balance, agg_func)

    print("processing prev table")

    ## Build Sum_Prev
    sum_prev = process_prev(prev, agg_func)
    
    print("final checks")
    #train.set_index('SK_ID_CURR')
    
    # check for nulls
    assert len(avg_bureau.isnull().any()[avg_bureau.isnull().any()==True].index) == 0
    assert len(sum_cc_balance.isnull().any()[sum_cc_balance.isnull().any()==True].index) == 0
    assert len(sum_payments.isnull().any()[sum_payments.isnull().any()==True].index) == 0
    assert len(sum_pc_balance.isnull().any()[sum_pc_balance.isnull().any()==True].index) == 0
    assert len(sum_prev.isnull().any()[sum_prev.isnull().any()==True].index) == 0

    print("process unified dataset")

    unified = process_unified(unified)

    print("merging feats into train and test - part1")

    feats = unified \
        .merge(avg_bureau, how='left', left_index=True, right_index=True) \
        .merge(sum_cc_balance, how='left', left_index=True, right_index=True) \
        .merge(sum_payments, how='left', left_index=True, right_index=True) \
        .merge(sum_pc_balance, how='left', left_index=True, right_index=True) \
        .merge(sum_prev, how='left', left_index=True, right_index=True) \

    print("extra feats")

    feats['DAYS_EMPLOYED'] = feats.DAYS_EMPLOYED.map(lambda x: np.nan if x == 365243 else x)
    feats['DAYS_EMPLOYED_PERC'] = np.sqrt(feats.DAYS_EMPLOYED / feats.DAYS_BIRTH)
    feats['INCOME_CREDIT_PERC'] = feats.AMT_INCOME_TOTAL / feats.AMT_CREDIT
    
    # need to debug this
    #train_feat['INCOME_PER_PERSON'] = np.log1p(train_feat.AMT_INCOME_TOTAL / train_feat.CNT_FAM_MEMBERS)

    print("feats done")

    # cleanup nulls
    feats_nulls = ['DAYS_EMPLOYED','OWN_CAR_AGE','MONTHS_BALANCE_mean_y','MONTHS_BALANCE_max_y','MONTHS_BALANCE_min_y','MONTHS_BALANCE_sum_y','MONTHS_BALANCE_std_y',
                   'AMT_BALANCE_mean','AMT_BALANCE_max','AMT_BALANCE_min','AMT_BALANCE_sum','AMT_BALANCE_std',
                   'AMT_CREDIT_LIMIT_ACTUAL_mean','AMT_CREDIT_LIMIT_ACTUAL_max','AMT_CREDIT_LIMIT_ACTUAL_min','AMT_CREDIT_LIMIT_ACTUAL_sum','AMT_CREDIT_LIMIT_ACTUAL_std',
                   'AMT_DRAWINGS_ATM_CURRENT_mean','AMT_DRAWINGS_ATM_CURRENT_max','AMT_DRAWINGS_ATM_CURRENT_min','AMT_DRAWINGS_ATM_CURRENT_sum','AMT_DRAWINGS_ATM_CURRENT_std',
                   'AMT_DRAWINGS_CURRENT_mean','AMT_DRAWINGS_CURRENT_max','AMT_DRAWINGS_CURRENT_min','AMT_DRAWINGS_CURRENT_sum','AMT_DRAWINGS_CURRENT_std',
                   'AMT_DRAWINGS_OTHER_CURRENT_mean','AMT_DRAWINGS_OTHER_CURRENT_max','AMT_DRAWINGS_OTHER_CURRENT_min','AMT_DRAWINGS_OTHER_CURRENT_sum','AMT_DRAWINGS_OTHER_CURRENT_std',
                   'AMT_DRAWINGS_POS_CURRENT_mean','AMT_DRAWINGS_POS_CURRENT_max','AMT_DRAWINGS_POS_CURRENT_min','AMT_DRAWINGS_POS_CURRENT_sum','AMT_DRAWINGS_POS_CURRENT_std',
                   'AMT_INST_MIN_REGULARITY_mean','AMT_INST_MIN_REGULARITY_max','AMT_INST_MIN_REGULARITY_min','AMT_INST_MIN_REGULARITY_sum','AMT_INST_MIN_REGULARITY_std',
                   'AMT_PAYMENT_CURRENT_mean','AMT_PAYMENT_CURRENT_max','AMT_PAYMENT_CURRENT_min','AMT_PAYMENT_CURRENT_sum','AMT_PAYMENT_CURRENT_std',
                   'AMT_PAYMENT_TOTAL_CURRENT_mean','AMT_PAYMENT_TOTAL_CURRENT_max','AMT_PAYMENT_TOTAL_CURRENT_min','AMT_PAYMENT_TOTAL_CURRENT_sum','AMT_PAYMENT_TOTAL_CURRENT_std',
                   'AMT_RECEIVABLE_PRINCIPAL_mean','AMT_RECEIVABLE_PRINCIPAL_max','AMT_RECEIVABLE_PRINCIPAL_min','AMT_RECEIVABLE_PRINCIPAL_sum','AMT_RECEIVABLE_PRINCIPAL_std',
                   'AMT_RECIVABLE_mean','AMT_RECIVABLE_max','AMT_RECIVABLE_min','AMT_RECIVABLE_sum','AMT_RECIVABLE_std',
                   'AMT_TOTAL_RECEIVABLE_mean','AMT_TOTAL_RECEIVABLE_max','AMT_TOTAL_RECEIVABLE_min','AMT_TOTAL_RECEIVABLE_sum','AMT_TOTAL_RECEIVABLE_std',
                   'CNT_DRAWINGS_ATM_CURRENT_mean','CNT_DRAWINGS_ATM_CURRENT_max','CNT_DRAWINGS_ATM_CURRENT_min','CNT_DRAWINGS_ATM_CURRENT_sum','CNT_DRAWINGS_ATM_CURRENT_std',
                   'CNT_DRAWINGS_CURRENT_mean','CNT_DRAWINGS_CURRENT_max','CNT_DRAWINGS_CURRENT_min','CNT_DRAWINGS_CURRENT_sum','CNT_DRAWINGS_CURRENT_std',
                   'CNT_DRAWINGS_OTHER_CURRENT_mean','CNT_DRAWINGS_OTHER_CURRENT_max','CNT_DRAWINGS_OTHER_CURRENT_min','CNT_DRAWINGS_OTHER_CURRENT_sum','CNT_DRAWINGS_OTHER_CURRENT_std',
                   'CNT_DRAWINGS_POS_CURRENT_mean','CNT_DRAWINGS_POS_CURRENT_max','CNT_DRAWINGS_POS_CURRENT_min','CNT_DRAWINGS_POS_CURRENT_sum','CNT_DRAWINGS_POS_CURRENT_std',
                   'CNT_INSTALMENT_MATURE_CUM_mean','CNT_INSTALMENT_MATURE_CUM_max','CNT_INSTALMENT_MATURE_CUM_min','CNT_INSTALMENT_MATURE_CUM_sum','CNT_INSTALMENT_MATURE_CUM_std',
                   'SK_DPD_mean_x','SK_DPD_max_x','SK_DPD_min_x','SK_DPD_sum_x','SK_DPD_std_x',
                   'SK_DPD_DEF_mean_x','SK_DPD_DEF_max_x','SK_DPD_DEF_min_x','SK_DPD_DEF_sum_x','SK_DPD_DEF_std_x',
                   'NUM_INSTALMENT_VERSION_mean','NUM_INSTALMENT_VERSION_max','NUM_INSTALMENT_VERSION_min','NUM_INSTALMENT_VERSION_sum','NUM_INSTALMENT_VERSION_std',
                   'NUM_INSTALMENT_NUMBER_mean','NUM_INSTALMENT_NUMBER_max','NUM_INSTALMENT_NUMBER_min','NUM_INSTALMENT_NUMBER_sum','NUM_INSTALMENT_NUMBER_std',
                   'DAYS_INSTALMENT_mean','DAYS_INSTALMENT_max','DAYS_INSTALMENT_min','DAYS_INSTALMENT_sum','DAYS_INSTALMENT_std',
                   'DAYS_ENTRY_PAYMENT_mean','DAYS_ENTRY_PAYMENT_max','DAYS_ENTRY_PAYMENT_min','DAYS_ENTRY_PAYMENT_sum','DAYS_ENTRY_PAYMENT_std',
                   'AMT_INSTALMENT_mean','AMT_INSTALMENT_max','AMT_INSTALMENT_min','AMT_INSTALMENT_sum','AMT_INSTALMENT_std',
                   'AMT_PAYMENT_mean','AMT_PAYMENT_max','AMT_PAYMENT_min','AMT_PAYMENT_sum','AMT_PAYMENT_std',
                   'PAYMENT_PERC_mean','PAYMENT_PERC_max','PAYMENT_PERC_min','PAYMENT_PERC_sum','PAYMENT_PERC_std',
                   'PAYMENT_DIFF_mean','PAYMENT_DIFF_max','PAYMENT_DIFF_min','PAYMENT_DIFF_sum','PAYMENT_DIFF_std',
                   'DPD_mean','DPD_max','DPD_min','DPD_sum','DPD_std', 'DBD_mean','DBD_max','DBD_min','DBD_sum','DBD_std',
                   'MONTHS_BALANCE_mean','MONTHS_BALANCE_max','MONTHS_BALANCE_min','MONTHS_BALANCE_sum','MONTHS_BALANCE_std',
                   'CNT_INSTALMENT_mean','CNT_INSTALMENT_max','CNT_INSTALMENT_min','CNT_INSTALMENT_sum','CNT_INSTALMENT_std',
                   'CNT_INSTALMENT_FUTURE_mean','CNT_INSTALMENT_FUTURE_max','CNT_INSTALMENT_FUTURE_min','CNT_INSTALMENT_FUTURE_sum','CNT_INSTALMENT_FUTURE_std',
                   'SK_DPD_mean_y','SK_DPD_max_y','SK_DPD_min_y','SK_DPD_sum_y','SK_DPD_std_y',
                   'SK_DPD_DEF_mean_y','SK_DPD_DEF_max_y','SK_DPD_DEF_min_y','SK_DPD_DEF_sum_y','SK_DPD_DEF_std_y',
                   'AMT_ANNUITY_mean','AMT_ANNUITY_max','AMT_ANNUITY_min','AMT_ANNUITY_sum','AMT_ANNUITY_std',
                   'AMT_APPLICATION_mean','AMT_APPLICATION_max','AMT_APPLICATION_min','AMT_APPLICATION_sum','AMT_APPLICATION_std',
                   'AMT_CREDIT_mean','AMT_CREDIT_max','AMT_CREDIT_min','AMT_CREDIT_sum','AMT_CREDIT_std',
                   'AMT_DOWN_PAYMENT_mean','AMT_DOWN_PAYMENT_max','AMT_DOWN_PAYMENT_min','AMT_DOWN_PAYMENT_sum','AMT_DOWN_PAYMENT_std',
                   'AMT_GOODS_PRICE_mean','AMT_GOODS_PRICE_max','AMT_GOODS_PRICE_min','AMT_GOODS_PRICE_sum','AMT_GOODS_PRICE_std',
                   'HOUR_APPR_PROCESS_START_mean','HOUR_APPR_PROCESS_START_max','HOUR_APPR_PROCESS_START_min','HOUR_APPR_PROCESS_START_sum','HOUR_APPR_PROCESS_START_std',
                   'NFLAG_LAST_APPL_IN_DAY_mean','NFLAG_LAST_APPL_IN_DAY_max','NFLAG_LAST_APPL_IN_DAY_min','NFLAG_LAST_APPL_IN_DAY_sum','NFLAG_LAST_APPL_IN_DAY_std',
                   'RATE_DOWN_PAYMENT_mean','RATE_DOWN_PAYMENT_max','RATE_DOWN_PAYMENT_min','RATE_DOWN_PAYMENT_sum','RATE_DOWN_PAYMENT_std',
                   'RATE_INTEREST_PRIMARY_mean','RATE_INTEREST_PRIMARY_max','RATE_INTEREST_PRIMARY_min','RATE_INTEREST_PRIMARY_sum','RATE_INTEREST_PRIMARY_std',
                   'RATE_INTEREST_PRIVILEGED_mean','RATE_INTEREST_PRIVILEGED_max','RATE_INTEREST_PRIVILEGED_min','RATE_INTEREST_PRIVILEGED_sum','RATE_INTEREST_PRIVILEGED_std',
                   'DAYS_DECISION_mean','DAYS_DECISION_max','DAYS_DECISION_min','DAYS_DECISION_sum','DAYS_DECISION_std',
                   'SELLERPLACE_AREA_mean','SELLERPLACE_AREA_max','SELLERPLACE_AREA_min','SELLERPLACE_AREA_sum','SELLERPLACE_AREA_std',
                   'CNT_PAYMENT_mean','CNT_PAYMENT_max','CNT_PAYMENT_min','CNT_PAYMENT_sum','CNT_PAYMENT_std',
                   'DAYS_FIRST_DRAWING_mean','DAYS_FIRST_DRAWING_max','DAYS_FIRST_DRAWING_min','DAYS_FIRST_DRAWING_sum','DAYS_FIRST_DRAWING_std',
                   'DAYS_FIRST_DUE_mean','DAYS_FIRST_DUE_max','DAYS_FIRST_DUE_min','DAYS_FIRST_DUE_sum','DAYS_FIRST_DUE_std',
                   'DAYS_LAST_DUE_1ST_VERSION_mean','DAYS_LAST_DUE_1ST_VERSION_max','DAYS_LAST_DUE_1ST_VERSION_min','DAYS_LAST_DUE_1ST_VERSION_sum','DAYS_LAST_DUE_1ST_VERSION_std',
                   'DAYS_LAST_DUE_mean','DAYS_LAST_DUE_max','DAYS_LAST_DUE_min','DAYS_LAST_DUE_sum','DAYS_LAST_DUE_std',
                   'DAYS_TERMINATION_mean','DAYS_TERMINATION_max','DAYS_TERMINATION_min','DAYS_TERMINATION_sum','DAYS_TERMINATION_std',
                   'APP_CREDIT_PERC_mean','APP_CREDIT_PERC_max','APP_CREDIT_PERC_min','APP_CREDIT_PERC_sum','APP_CREDIT_PERC_std','DAYS_EMPLOYED_PERC']

    for col in feats_nulls:
        feats[col] = feats[col].fillna(0)

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
