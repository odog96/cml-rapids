# simple feature engineering from A_First_Model notebook in script form
import cudf

def see_percent_missing_values(df):
    """

    reads in a dataframe and returns the percentage of missing data

    Args:
        df (dataframe): the dataframe that we are analysing
    Returns:
        percent_missing (dataframe): a dataframe with percentage missing for filtering

    """
    
    total_missing = df.isnull().sum()/df.shape[0]
    percent_missing = total_missing*100
    return percent_missing.sort_values(ascending=False).round(1)


def basic_feature_engineering(train, test, gpu=False):
    """

    reads in a train and test set of data and processes as per the basic
    feature engineering example

    Args:
        train (dataframe): the training dataframe (should include TARGET)
        test (dataframe): the testing dataframe
        gpu (boolean): whether to use cudf or not

    Returns:
        train (dataframe): the processed train frame
        test (dataframe): the processed test frame
        train_target (dataframe): The training target column

    """

    if gpu:
        import cudf as dd
    else:
        import pandas as dd

    app_train_mis_values = see_percent_missing_values(train)
    df_app_train_miss_values= dd.DataFrame({'columns': app_train_mis_values.index, 
                                        'missing percent': app_train_mis_values.values})

    if type(df_app_train_miss_values) == cudf.core.dataframe.DataFrame:
        drop_columns = df_app_train_miss_values[df_app_train_miss_values['missing percent'] \
                                        >= 40]['columns'].to_arrow().to_pylist()
    else:
        drop_columns = df_app_train_miss_values[df_app_train_miss_values['missing percent'] \
                                        >= 40]['columns'].tolist()

    train = train.drop(drop_columns, axis=1)
    test = test.drop(drop_columns, axis=1)
    train_target = train['TARGET']
    train = train.drop('TARGET', axis=1)
    # here we will use a basic dummy treatment
    # we merged the dataframes first because when we dummify 
    # we could have some columns only in train or only in test. Merging first will prevent this 
    unified = dd.concat([train, test])
    dummy_cols = unified.select_dtypes(['bool', 'O', 'category']).columns.tolist()
    unified = dd.get_dummies(unified, columns=dummy_cols, dtype='int64')

    # XGB for pandas does not like Int64
    for col in unified.select_dtypes('Int64').columns.tolist():
        unified[col] = unified[col].fillna(int(unified[col].mean()))
        unified[col] = unified[col].astype('int64')

    for col in unified.isna().any()[unified.isna().any()==True].index.to_arrow().tolist():
        unified[col] = unified[col].fillna(0)

    train = unified[0:307511]
    test = unified[307511:]

    return train, test, train_target
