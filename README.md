#CLASSIFICATION ASSESSMENT OF TUMOR SUBTYPES

This repository contains an R implementation of a multinomial classifier developed to predict breast cancer subtypes (HER2+, HR+, Triple Negative) from array-CGH data.

The goal of this project was to build a reproducible machine learning pipeline for high dimensional genomic data, including feature selection, nested cross validation and hyperparameter optimisation using glmnet.

The analysis script is available in the "code" directory and the summary outputs of the repeated nested cross-validation experiments are provided in "results".

The dataset used in this project was distributed as part of a university course and cannot be shared publicly.

Current extensions of the work include testing the effect of larger feature subsets and implementing a logistic regression baseline model for comparison.
