House Price Prediction  - Advanced Regression Techniques

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Acknowledgments

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.

Steps deployed

  1.Fetched Dataset from kaggle

  2.Did initial preprocessing by handling missing values

  3.Extensive EDA is done to obtain relationships between different features and to get other insights

  4.Categorical Values are handled according to the ordinality

  5.Ordered categories are label encoded and others are One hot encoded.

  6.Dimensionality reduction is done.

  7.modelling - Lasso, ElasticNet, Ridge along with XGboost is applied

  8.Hypertuning is performed

  9.Best model is used to create submission file

RMSE score is 1.26.
Future work is to apply stacked regressors to improve accuracy
