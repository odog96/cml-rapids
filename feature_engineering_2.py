### More Complicated Data Engineering Job
## From https://www.kaggle.com/tunguz/xgb-simple-features/comments

import cudf
import numpy as np
import gc

def one_hot_encoder(dd, df, nan_as_category = True):
    # one hot encode function for the data
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if (df[col].dtype in ['object']) ]
    
    df[categorical_columns] = df[categorical_columns].astype('O') # there is issue with categorical
    df = dd.get_dummies(df, columns=categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def process_unified(unified, dd):

    # this wont work in dask will need to adjust
    income_by_organisation = unified[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    
    unified['DAYS_EMPLOYED'] = unified['DAYS_EMPLOYED'].map(lambda x: np.nan if x == 365243 else x)

    docs = [_f for _f in unified.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in unified.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    unified['NEW_CREDIT_TO_ANNUITY_RATIO'] = unified['AMT_CREDIT'] / unified['AMT_ANNUITY']
    unified['NEW_CREDIT_TO_GOODS_RATIO'] = unified['AMT_CREDIT'] / unified['AMT_GOODS_PRICE']
    
    # kurtosis works differently in cudf and hence we had to add an exception
    # for if we are feeding in pandas format data or not
    if type(unified) == cudf.core.dataframe.DataFrame:
        unified['NEW_DOC_IND_KURT'] = unified[docs].to_pandas().kurtosis(axis=1)
    else:
        unified['NEW_DOC_IND_KURT'] = unified[docs].kurtosis(axis=1)
    
    unified['NEW_LIVE_IND_SUM'] = unified[live].sum(axis=1)
    unified['NEW_INC_PER_CHLD'] = unified['AMT_INCOME_TOTAL'] / (1 + unified['CNT_CHILDREN'])
    unified['NEW_INC_BY_ORG'] = unified['ORGANIZATION_TYPE'].map(income_by_organisation)
    unified['NEW_EMPLOY_TO_BIRTH_RATIO'] = unified['DAYS_EMPLOYED'] / unified['DAYS_BIRTH']
    unified['NEW_ANNUITY_TO_INCOME_RATIO'] = unified['AMT_ANNUITY'] / (1 + unified['AMT_INCOME_TOTAL'])
    unified['NEW_SOURCES_PROD'] = unified['EXT_SOURCE_1'] * unified['EXT_SOURCE_2'] * unified['EXT_SOURCE_3']
    unified['NEW_EXT_SOURCES_MEAN'] = unified[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    unified['NEW_SCORES_STD'] = unified[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    unified['NEW_SCORES_STD'] = unified['NEW_SCORES_STD'].fillna(unified['NEW_SCORES_STD'].mean().astype('float32'))
    unified['NEW_CAR_TO_BIRTH_RATIO'] = unified['OWN_CAR_AGE'] / unified['DAYS_BIRTH']
    unified['NEW_CAR_TO_EMPLOY_RATIO'] = unified['OWN_CAR_AGE'] / unified['DAYS_EMPLOYED']
    unified['NEW_PHONE_TO_BIRTH_RATIO'] = unified['DAYS_LAST_PHONE_CHANGE'] / unified['DAYS_BIRTH']
    unified['NEW_PHONE_TO_BIRTH_RATIO'] = unified['DAYS_LAST_PHONE_CHANGE'] / unified['DAYS_EMPLOYED']
    unified['NEW_CREDIT_TO_INCOME_RATIO'] = unified['AMT_CREDIT'] / unified['AMT_INCOME_TOTAL']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        unified[bin_feature], uniques = dd.factorize(unified[bin_feature])

    # Categorical features with One-Hot encode
    unified, cat_cols = one_hot_encoder(dd, unified, nan_as_category=True)

    return unified

def process_bureau_and_balance(bureau, bureau_balance, dd, nan_as_category = True):
    
    bb, bb_cat = one_hot_encoder(dd, bureau_balance, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(dd, bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = dd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.merge(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = dd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Bureau: Active credits - using only numerical aggregations
    # note this works only when the dtype of the CREDIT_ACTIVE was object and not category
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = dd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.merge(active_agg, how='left', on='SK_ID_CURR')

    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = dd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.merge(closed_agg, how='left', on='SK_ID_CURR')

    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

def process_previous_applications(prev, dd, nan_as_category = True):
    
    prev, cat_cols = one_hot_encoder(dd, prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = dd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = dd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.merge(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = dd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.merge(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

def pos_cash(pos, dd, nan_as_category = True):

    pos, cat_cols = one_hot_encoder(dd, pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = dd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

def installments_payments(ins, dd, nan_as_category = True):
    ins, cat_cols = one_hot_encoder(dd, ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    
    if type(ins) == cudf.core.dataframe.DataFrame:
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    else:
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    #prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': [ 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': [ 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = dd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

def credit_card_balance(cc, dd, nan_as_category = True):
    cc, cat_cols = one_hot_encoder(dd, cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    # Identify categorical columns
    categorical_columns = cc.select_dtypes(include='category').columns

    # Drop categorical columns
    cc.drop(categorical_columns, axis=1, inplace = True)

    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = dd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg