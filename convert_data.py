# this script reads the data into a parquet format.
# we specify types to make the data "cleaner"

import pandas as dd

print("converting bureau balance")
bureau_balance = dd.read_csv('data/bureau_balance.csv')
bureau_balance['STATUS'] = bureau_balance.STATUS.astype('category')
bureau_balance.to_parquet('raw_data/bureau_balance.parquet')
## Links to Bureau on sK_ID_BUREAU

print("converting bureau")
bureau = dd.read_csv('data/bureau.csv',
                    dtype={'CREDIT_ACTIVE': 'category', 'CREDIT_CURRENCY': 'category'})
bureau.to_parquet('raw_data/bureau.parquet')
## Links to Train data on SK_ID_CURR

print("converting credit card balance")
cc_balance = dd.read_csv('data/credit_card_balance.csv',
                        dtype={'NAME_CONTRACT_STATUS': 'category'})
cc_balance.to_parquet('raw_data/cc_balance.parquet')
## Links to Prev on SK_ID_PREV
## Though also have SK_ID_CURR

print("converting installments payments")
payments = dd.read_csv('data/installments_payments.csv')
payments.to_parquet('raw_data/payments.parquet')
## Links to Prev on SK_ID_PREV
## Though also have SK_ID_CURR

print("converting POS CASH Balance")
pc_balance = dd.read_csv('data/POS_CASH_balance.csv')
pc_balance.to_parquet('raw_data/pc_balance.parquet')
## Links to Prev on SK_ID_PREV
## Though also have SK_ID_CURR

print("converting prev applications")
prev = dd.read_csv('data/previous_application.csv',
                  dtype={'NAME_CONTRACT_TYPE': 'category', 'WEEKDAY_APPR_PROCESS_START': 'category',
                        'FLAG_LAST_APPL_PER_CONTRACT': 'category', 'NAME_CONTRACT_STATUS': 'category',
                        'NAME_SELLER_INDUSTRY': 'category', 'NAME_YIELD_GROUP': 'category'})
# 'NFLAG_INSURED_ON_APPROVAL': 'bool' there are some na in here we need to handle too                        
prev.to_parquet('raw_data/prev.parquet')
## Previous loans with Home Credit Group

print("converting train and test")
train_test_dtype_dict = {'NAME_CONTRACT_TYPE': 'category', 'CODE_GENDER': 'category',
                          'NAME_INCOME_TYPE': 'category',
                          'NAME_EDUCATION_TYPE': 'category', 'NAME_FAMILY_STATUS': 'category',
                          'NAME_HOUSING_TYPE': 'category', 'FLAG_MOBIL': 'bool', 
                          'FLAG_EMP_PHONE': 'bool', 'FLAG_WORK_PHONE': 'bool',
                          'FLAG_CONT_MOBILE': 'bool', 'FLAG_PHONE': 'bool', 'FLAG_EMAIL': 'bool',
                          'CNT_FAM_MEMBERS': 'Int64', 'REGION_RATING_CLIENT': 'category',
                          'REGION_RATING_CLIENT_W_CITY': 'category', 'WEEKDAY_APPR_PROCESS_START': 'category',
                          'HOUR_APPR_PROCESS_START': 'category', 'REG_REGION_NOT_LIVE_REGION': 'bool',
                          'REG_REGION_NOT_WORK_REGION': 'bool', 'LIVE_REGION_NOT_WORK_REGION': 'bool',
                          'REG_CITY_NOT_LIVE_CITY': 'bool', 'REG_CITY_NOT_WORK_CITY': 'bool',
                          'LIVE_CITY_NOT_WORK_CITY': 'bool', 'ORGANIZATION_TYPE': 'category',
                          'OBS_30_CNT_SOCIAL_CIRCLE': 'Int64', 'DEF_30_CNT_SOCIAL_CIRCLE': 'Int64',
                          'OBS_60_CNT_SOCIAL_CIRCLE': 'Int64', 'DEF_60_CNT_SOCIAL_CIRCLE': 'Int64',
                          'DAYS_LAST_PHONE_CHANGE': 'Int64', 'FLAG_DOCUMENT_2': 'bool',
                          'FLAG_DOCUMENT_3': 'bool', 'FLAG_DOCUMENT_4': 'bool', 'FLAG_DOCUMENT_5': 'bool',
                          'FLAG_DOCUMENT_6': 'bool', 'FLAG_DOCUMENT_7': 'bool', 'FLAG_DOCUMENT_8': 'bool',
                          'FLAG_DOCUMENT_9': 'bool', 'FLAG_DOCUMENT_10': 'bool', 'FLAG_DOCUMENT_11': 'bool',
                          'FLAG_DOCUMENT_12': 'bool', 'FLAG_DOCUMENT_13': 'bool', 'FLAG_DOCUMENT_14': 'bool', 
                          'FLAG_DOCUMENT_15': 'bool', 'FLAG_DOCUMENT_16': 'bool', 'FLAG_DOCUMENT_17': 'bool',
                          'FLAG_DOCUMENT_18': 'bool', 'FLAG_DOCUMENT_19': 'bool', 'FLAG_DOCUMENT_20': 'bool',
                          'FLAG_DOCUMENT_21': 'bool', 'AMT_REQ_CREDIT_BUREAU_HOUR': 'Int64',
                          'AMT_REQ_CREDIT_BUREAU_DAY': 'Int64', 'AMT_REQ_CREDIT_BUREAU_WEEK': 'Int64',
                          'AMT_REQ_CREDIT_BUREAU_MON': 'Int64', 'AMT_REQ_CREDIT_BUREAU_QRT': 'Int64',
                          'AMT_REQ_CREDIT_BUREAU_YEAR': 'Int64'}

train = dd.read_csv('data/application_train.csv',
                    index_col='SK_ID_CURR',
                    dtype=train_test_dtype_dict)
train.FLAG_OWN_CAR = train.FLAG_OWN_CAR.eq('Y').mul(1).astype('bool')
train.FLAG_OWN_REALTY = train.FLAG_OWN_REALTY.eq('Y').mul(1).astype('bool')
train.to_parquet('raw_data/train.parquet')

test = dd.read_csv('data/application_test.csv',
                    index_col='SK_ID_CURR',
                    dtype=train_test_dtype_dict)
test.FLAG_OWN_CAR = test.FLAG_OWN_CAR.eq('Y').mul(1).astype('bool')
test.FLAG_OWN_REALTY = test.FLAG_OWN_REALTY.eq('Y').mul(1).astype('bool')
test.to_parquet('raw_data/test.parquet')

print("done")