### More Complicated Data Engineering Job
## From https://www.kaggle.com/tunguz/xgb-simple-features/comments

import cudf
import numpy as np
import gc

def one_hot_encoder(dd, df, nan_as_category = True):
    # add an exclusion for 
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
    
    # debug for cudf
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
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
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
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = dd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg