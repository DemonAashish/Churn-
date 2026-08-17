
Chrun data analysis

Dataset loaded with 1000 records and 21 columns.

--- Logistic Regression ---
Accuracy: 0.8550
              precision    recall  f1-score   support

           0       0.89      0.83      0.85       103
           1       0.83      0.89      0.86        97

    accuracy                           0.85       200
   macro avg       0.86      0.86      0.85       200
weighted avg       0.86      0.85      0.85       200


--- Random Forest ---
Accuracy: 0.9250
              precision    recall  f1-score   support

           0       0.95      0.90      0.93       103
           1       0.90      0.95      0.92        97

    accuracy                           0.93       200
   macro avg       0.93      0.93      0.92       200
weighted avg       0.93      0.93      0.93       200

/opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [22:04:23] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:794: 
Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)

--- XGBoost ---
Accuracy: 0.9350
              precision    recall  f1-score   support

           0       0.94      0.93      0.94       103
           1       0.93      0.94      0.93        97

    accuracy                           0.94       200
   macro avg       0.93      0.94      0.93       200
weighted avg       0.94      0.94      0.94       200


Best Model: XGBoost with Accuracy: 0.9350
