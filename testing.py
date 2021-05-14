print("loading cudf library")
import cudf
print("loaded cudf library")


## Simple hello world script to verify that everything is working

train_path = "data/application_train.csv"

print("reading data with rapids")
tips_df = cudf.read_csv(train_path)

print("the dataframe is shaped: {0}".format(tips_df.shape))

print("Completed Test")