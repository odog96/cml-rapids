## Feature Engineering using dask

import time

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask_cudf
from feature_engineering import feature_engineering

cluster = LocalCUDACluster()
client = Client(cluster)
client

### Load Data
bureau_balance = dask_cudf.read_csv('data/bureau_balance.csv')
bureau = dask_cudf.read_csv('data/bureau.csv')
cc_balance = dask_cudf.read_csv('data/credit_card_balance.csv')
payments = dask_cudf.read_csv('data/installments_payments.csv')
pc_balance = dask_cudf.read_csv('data/POS_CASH_balance.csv')
prev = dask_cudf.read_csv('data/previous_application.csv')
train = dask_cudf.read_csv('data/application_train.csv')
test = dask_cudf.read_csv('data/application_test.csv')

feature_engineering(bureau_balance, bureau, cc_balance, payments, pc_balance,
                        prev, train, test)

client.close()