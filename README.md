# predicting-county-COVID-19-deaths

Collaborators: Richard Chang and John Greer

Note, the full project write-up can be found at Final Project (RCC JG).pdf


## Description 
State and local health agencies are increasingly reliant on statistical measures to direct COVID-19 resource allocation towards populations and geographic areas. However, the predictive value of these indicators is inherently limited by their construction, which occurred before COVID-19 and is used to predict or evaluate conditions that may only be peripherally related to COVID-19.We attempt to construct a COVID-19 death predictor utilizing the most granular national data publicly available, at the county level, using measures that combine socio-economic and health indicators. Various machine learning models are applied and evaluated. Our results indicate that ridge regression generally  produces the lowest error rates although error rates vary drastically based on the random test-train split. 

## Data 
The features utilized were sourced from the Census Bureau’s 2014-2018 American Community Survey and Robert Wood Johnson Foundation’s County Health Rankings. The target labels, COVID-19 related deaths, came from the Centers for Disease Control and Prevention’s Provisional 
COVID-19 Death Counts in the United States by County. 

## Methods 
For this project, we wrote all functions across the machine learning pipeline from scratch, using Python and Numpy. Our chosen models for this project were all variations on linear regressoion, ranging from the vanilla version to ridge regression and polynomial expansion. We also employed k-folds cross-validation and grid search functions. 

Our implemention of ml functions can be found in Utils.py and ml_helpers.py. Our data cleaning occurs in 01_acs_data_consolidation.ipynb, and our application of ML models and subsequent analysis occurs in 02_Data Analysis.ipynb


