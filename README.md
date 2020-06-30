# kaggle_toxic_comment
Kaggle Toxic Comment Competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

In this NLP project, we are provided with wiki comment data with multi-lables, the objective is to predict the probability of each lable and the result will be judged based on ROC AUC. 

Our team's contributions are：
1. comprehensively explored data  
2. created a util script for sharing among team members
3. comprehensive feature engineering:
  - word embedding
  - sentiment analysis
  - emoji detection
  - language detection
  
4. used few machine learning models with different multi_lable prediction techniques:
  * models:
  - naive bayes
  - random forest
  - svm
  - lightgbm
  
  * wrappers:
  - one vs rest
  - chain classifier
 
5. compiled a class that will be able to automatically run models with different combination of model and wrappers to simplify model selection
