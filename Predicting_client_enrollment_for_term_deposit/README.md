# Banking Marketing

### Find the best strategies to improve for the next marketing campaign. How can the financial institution have a greater effectiveness for future marketing campaigns? In order to answer this, we have to analyze the last marketing campaign the bank performed and identify the patterns that will help us find conclusions in order to develop future strategies.


_Parth Pandya_

---
**Source**: https://archive.ics.uci.edu/ml/datasets/bank+marketing#


**Procedure for building model: CRISP-DM (Cross-Industry Standard Procedure for Data Mining)**

* **Business Understanding** : Predicting client enrollment for a term deposit to plan marketing campaign

* **Data Understanding and Preparation**: 
  * 4190 rows - 10 cateogorical, 11 numerical features
  * one feature 'pdays' heavily skewed
  * Detect outliers and impute missing values 
  * Min-max normalization
  * Dummy coding categorical features
  * Implement PCA for dimensional reduction
  * Collinearity between features to be excluded
  
* **Machine learning Models**: knn, logistic, SVM, Neural Network, SVM, Rule Learner(Separate and Conquer Approach), Random Forests, Ensembled models (Bagging, Boosting), Stacked Ensembled (svm, knn)

* Model evaluation metrics: **kappa statistic, AUC**

**Best Model: Rule Learner: AUC: 0.80, kappa: 0.7**





