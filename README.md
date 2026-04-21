# multinomial-classifier-arrayCGH-glmnet
Classification Assessment of Tumor Subtypes using Elastic Net (glmnet)
The workflow includes:

- feature selection using filterVarImp
- nested cross-validation
- hyperparameter tuning (alpha, lambda)
- model evaluation across repetitions
- prediction on independent validation dataset

The classifier was implemented in R using caret and glmnet.
## Ongoing improvements

Planned extensions of this work include:

- testing the classifier with larger feature subsets to evaluate robustness with respect to feature selection size
- implementing a baseline logistic regression model to benchmark performance against the elastic net approach
