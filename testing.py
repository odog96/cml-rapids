import cudf

## Simple hello world script to verify that everything is working

train_path = "data/application_train.csv"
tips_df = cudf.read_csv(train_path)
print("the dataframe is shaped: {0}".format(tips_df.shape))