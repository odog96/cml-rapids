# RAPIDS.ai on Cloudera CML

In this article, we will cover leveraging RAPIDS to accelerate your machine learning projects on Cloudera's CML Platform

## Introduction

In our previous article, we walked through the steps of installing and running deep learning models with CML NVIDIA runtimes.
In this article, we will cover the RAPIDS runtime and show you how you can leverage NVIDIA RAPIDs to accelerate your non deeplearning experiments on Cloudera CML

## What is RAPIDs

Rapids is a series of libraries from NVIDIA that bring the power of GPU compute to standard Data Science operations, be they exploratory data analysis, through to feature engineering, model building and visualisation. For more information see: <RAPIDs link>

## Scenario

In this tutorial, I will use the Kaggle Home Credit Default Risk dataset. See: https://www.kaggle.com/c/home-credit-default-risk/overview 
This is typical classification machine learning problem. We will extract and transform the data using RAPIDS.ai libraries and build our GPU Accelerated CuML / Xgboost model.

## TODOS

- get the data munging into an etl process to write out the files processed files and save on the VRAM
- can also explore porting to dask-cudf
