# Module-12-Assignment - Credit Risk Classification

#### by Alex Melino

#

## Overview of the Analysis

This assignment covers supervised machine learning, and applies modelling techniques to solve a credit classification problem. The dataset used (found in the 'Resources' folder of this repo as 'lending_data.csv') consists of historical lending activity from a peer-to-peer lending services company, and this data was used to build models that can assist in identifying the creditworthiness of borrowers.

The assignment (found in this repo as 'credit_risk_resampling.ipynb') begins with reading the data and splitting it into training and testing sets. These sets were then used to create a logistic regression mdoel which provided a prediction of credit worthiness. The results were analyzed by assessing accuracy, precision, and recall scores.

The next section employed the technique of re-sampling the training data (in this case, specifically, the over-sampling method) in order to try and obtain more accurate and precise predictions. The data was re-sampled and was once again applied to a logistic regression model. The results were then compared to the original logistic regression model to determine which one performed better at making creditworthiness predictions.

The libraries and dependencies used in this analysis are numpy, pandas, pathlib, sklearn, and imblearn.


## Results


* Machine Learning Model 1 - Logistic Regression:
  * Balanced Accuracy: 95.2%
  * Precision: 85%
  * Recall: 91%
#
* Machine Learning Model 2 - Logistic Regression with Oversampled Data:
  * Balanced Accuracy: 99.4%
  * Precision: 84%
  * Recall: 99%



## Summary

In summary, the model with the oversampled data performed slightly worse when identifying true positives (healthy loans), mis-identifying 14 more healthy loans as high-risk than with the base sample data (116 false negatives compared to 102 with the base model). But this was statistically insignificant when looking at the number of samples of true positives overall (18765). 

The oversampled data model perfomed significantly better when identifying high-risk loans, and in this example that is the more important situation to identify. It was able to identify 99% of the high-risk loans compared to 91% when using the base sample data (4 false positives compared to 56 with the base model). Therefore, the logistic regression model fit with oversampled data performed better overall than the model fit with the base sample data.

