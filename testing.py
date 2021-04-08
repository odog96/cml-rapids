import cudf

train_path = "data/application_train.csv"

tips_df = cudf.read_csv(train_path)

print(tips_df.shape)