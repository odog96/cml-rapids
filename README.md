# RAPIDS.ai on Cloudera CML

In this article, we will cover leveraging RAPIDS to accelerate your machine learning projects on Cloudera's CML Platform

## Introduction

In our previous article, we walked through the steps of installing and running deep learning models with CML NVIDIA runtimes.
In this article, we will cover the RAPIDS runtime and show you how you can leverage NVIDIA RAPIDs to accelerate your non deeplearning experiments on Cloudera CML

## What is RAPIDs

Rapids is a series of libraries from NVIDIA that bring the power of GPU compute to standard Data Science operations, be they exploratory data analysis, feature engineering or model building. For more information see: <RAPIDs link>. The primary components that we will use today will be CuDF, cuML and GPU Acclerated Xgboost. CuDF is a drop in replacement for pandas that provides GPU Accelerated data exploration and transformation tools. cuML is a replacement for scikit-learn that provides the same familiar API with GPU accelerated algorithms. XGBoost, whilst not officially part of the RAPIDs stack is a famous algorithm for standard Machine Learning tasks that can be greatly accelerated by GPU Compute.

## Scenario

In this tutorial, I will use the Kaggle Home Credit Default Risk dataset. See: https://www.kaggle.com/c/home-credit-default-risk/overview 
This is typical classification machine learning problem. The Home Credit Default Risk problem is about predicting the chance that a customer will default on a loan, a common financial services industry problem set. The try and predict this, an extensive dataset including anonymised details on the individual loanee and their historical credit history including time series information on the rate at which they repaid historical loans.

For our exercise, this dataset provides an extensive set of data including multiple tables that need to be joined back together to produce a featureset for machine learning. In this worked example, we will extract and transform the data using RAPIDS.ai libraries and build our GPU Accelerated CuML / Xgboost model.

The focus of this tutorial will be more on the mechanics of leveraging the RAPIDs library and not on the "best" model. To see more information on the winning submission See: https://www.kaggle.com/c/home-credit-default-risk/discussion/64821

## Get the Dataset
### Data Ingestion

The raw data is in a series of CSV files. Exploring the dataset, there are numerical columns, categorical and boolean columns. 
### Feature Engineering

### Modelling

### Accessing Models
## TODOS

- can also explore porting to dask-cudf
