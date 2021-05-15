# RAPIDS.ai on Cloudera CML

In this article, we will cover leveraging RAPIDS to accelerate your machine learning projects on Cloudera's CML Platform

## Introduction

In the previous blog post in this series, we walked through the steps for leveraging Deep Learning in your Cloudera Machine Learning (CML) projects. This year, we expanded our partnership with NVIDIA, enabling your data teams to dramatically speed up compute processes for data engineering and data science workloads with no code changes using RAPIDS AI. RAPIDS on the Cloudera Data Platform comes pre-configured with all the necessary libraries and dependencies to bring the power of RAPIDS to your projects.

## What is RAPIDs

RAPIDS brings the power of GPU compute to standard Data Science operations, be it exploratory data analysis, feature engineering or model building. For more information see: <https://rapids.ai/>. The RAPIDS libraries are designed as drop-in replacements for common Python data science libraries like pandas (cuDF), numpy (cuPy), sklearn (cuML) and dask (dask_cuda). By leveraging the parallel compute capacity of GPUs the time for complicated data engineering and data science tasks can be dramatically reduced, accelerating the timeframes for Data Scientists to take ideas from concept to production.
## Scenario

In this tutorial, we will illustrate how RAPIDS can be used to tackle the Kaggle Home Credit Default Risk challenge. The Home Credit Default Risk problem is about predicting the chance that a customer will default on a loan, a common financial services industry problem set. To try and predict this, an extensive dataset including anonymised details on the individual loanee and their historical credit history are included. See https://www.kaggle.com/c/home-credit-default-risk/overview for more details.

As a machine learning problem, it is a classification task with tabular data, a perfect fit for RAPIDs.

The focus of this tutorial will be on the mechanics of leveraging the RAPIDs library and not on building the best performing model for the leaderboard. To see more information on the winning submission See: https://www.kaggle.com/c/home-credit-default-risk/discussion/64821

## Project Setup

To follow along, clone the repo at: https://github.com/Data-drone/cml_rapids.git into a new CML Project. 
![New Project From Git](images/CreateProject.png)

In this example we will use a Jupyter Notebook session to run our code. Create a session with 8 cores, 16GB memory and 1 GPU

Install the requirements from a terminal session with:
```bash
pip install -r requirements.txt
```

## Get the Dataset

For the code to work, the data in it's CSV format should be placed into the data subfolder. The dataset can be downloaded from: https://www.kaggle.com/c/home-credit-default-risk/data
![data_subfolder](images/data_folder.png)

To validate that our image is working and that RAPIDS is correctly configured, run `testing.py` from a terminal session in jupyterlab.
![testing_RAPIDS](images/testing_rapids.gif)

The script will go through loading RAPIDs libraries then leveraging them to load and processing a datafile.

Common problems at this stage can be related to GPU versions. RAPIDS is only supported on newer NVIDIA gpus. For AWS this means at least P3 instances. P2 GPU instances are not supported.
![GPU_error_message](images/old_gpu.png)
### Data Ingestion

The raw data is in a series of CSV files. We will firstly convert this to parquet format as most data lakes exist as object stores full of parquet files. Parquet also stores datatypes which makes reading back and processing the files later slightly easier.

Run the `convert_data.py` script. This will open the csvs with correctly data types then save them out as parquet in the `raw_data` folder. 
![running_convert_data](images/convert_data.gif)

Now we have all our parquet datasets to continue on our RAPIDS journey
![folder_structure_afterwards](images/processed_data.png)

Exploring the dataset, there are numerical columns, categorical and boolean columns. The `application_test` and `application_train` files contain the main features that we will be building our model off of whilst the other tables provide some supplementary data. Feel free to skim through: `First_Exploration.ipynb` in order to see some basic exploration of the datasets. 

In the `First_Exploration.ipynb` we also leverage `cuXfilter`, a RAPIDS-accelerated cross filtering visualization library for some of the charts.

### Simple Exploration and Model

As with all Machine Learning problems, lets start with a simple model. This gives us the opportunity to build a baseline to improve off of and also to check that Machine Learning can learn something off the data right off the bat.

Open `A_First_Model.ipynb`

At the start of this notebook, you can choose which set of libraries to load.
The RAPIDs set or the Pandas set. Just run one of these cells.
![choose_which_set](images/Choose_version.gif)

This notebook goes through loading just the train and test datasets
![train_test_cell](images/load_data.png)

Some simple filtering out of columns with a lot of missing values
![missing_columns](images/Check_Missing.gif)

It is worth noting that although RAPIDS `cudf` is mostly a drop in replacement for `pandas`, we do need to change some parts to make it work seamlessly.

```python
if type(df_app_train_miss_values) == cudf.core.dataframe.DataFrame:
    drop_columns = df_app_train_miss_values[df_app_train_miss_values['missing percent'] \
                                        >= 40]['columns'].to_arrow().to_pylist()
else:
    drop_columns = df_app_train_miss_values[df_app_train_miss_values['missing percent'] \
                                        >= 40]['columns'].tolist()
```

The training of the model
![missing_columns](images/Training_Model_Jupyter.gif)

And analysing the results.
![model_analysis](images/Feature_Importances.png)

From our testing, the RAPIDS accelerated pipeline is ~28% faster.
### Feature Engineering

Now that we have a feel for how this works, lets look at a more advanced feature engineering pipeline.
For our simple feature engineering pipeline, we only used the main training table and didn't look at the other tables in the dataset.

For our advanced feature engineering pipeline, we will include the auxiliary data and also engineering some additional features.

Open the Comparing_Frameworks.ipynb file to see compare how cudf and pandas compare.

    NOTE: The function for engineering the features have been written to be compatible with Pandas and cuDF and can be found in `feature_engineering_2.py` 

The notebook is split into two sections. RAPIDS cuDF and Pandas.
From our testing, we see the following in terms of performance:

| Process        | RAPIDS (wall time) | Pandas (wall time)  |
| ------------- |:-------------:| :-----:|
| Ingest Data      | 5 secs | 9.83 secs |
| Generate Features      | 21.3 secs | 68.1 secs |
| Write Data | 5.64 secs | 9.8 secs |

This is based on a P3 Worker with 8 Cores and 16 GB RAM.

We can see that for all parts of the process, RAPIDs offers higher performance than raw Pandas. It is worth noting at this stage, that RAPIDs cuDF can only take advantage of one GPU. Should we wish to scale beyong a single GPU, we will need to leverage `dask_cudf`.
### Modelling

For the advanced modelling section, we will again leverage xgboost as our primary method. To enable GPU Acceleration, we set the `tree_method` to `gpu_hist`. That is really all we need to do to leverage GPU compute!

![gpu_hist](images/gpu_hist.png)


With the Home Credit Default Risk Challenge, overfitting is very easy. So we have included a cross validation step here. In order to use `train_test_split` with RAPIDS cudf frames, we use the `cuml` version instead. cuML, however, doesn't have `StratifiedKFold` sampling so we will use the `sklearn` version.

`StratifiedKFold` isn't very computationally expensive however so it doesn't matter that we aren't running this on GPU. The resulting indexes can also be used directly with cudf dataframes via `iloc` as per normal.

![KFold_Training](images/KFold_Training.gif)

### Assessing Models

With our model trained, we can have a look at the confusion matrix and auc scores from our model. Again, we use cuml versions so that we don't have to transfer the data back to CPU.

![Results](images/Results.png)

xgboost also features gpu accelerated feature importance calculations and shap calculations for explanability. For a full explanation of shap values see: https://www.kaggle.com/dansbecker/shap-values 

![Shap_values](images/adv_model_explainability.png)
### Next Steps

The following post is part 3 of the series, GPUs on CML. If you would like to learn more about how you can leverage RAPIDS to accelerate your Machine Learning Projects in Cloudera Machine Learning, be sure to check out part 1<https://blog.cloudera.com/enabling-nvidia-gpus-to-accelerate-model-development-in-cloudera-machine-learning/> & part 2<https://blog.cloudera.com/deep-learning-with-nvidia-gpus-in-cloudera-machine-learning/> of the blog series.