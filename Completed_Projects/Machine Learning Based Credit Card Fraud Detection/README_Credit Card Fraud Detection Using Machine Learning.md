# Credit Card Fraud Detection Using Machine Learning
# Problem It Solves / Context
Credit card fraud leads to significant financial losses and customer frustration due to static detection systems that either miss fraud or generate excessive false positives. This project explores using ensemble machine learning models to more effectively distinguish fraudulent transactions from legitimate ones.
# Tools / Technologies Used
- Python (pandas, scikit-learn, XGBoost, imbalanced-learn)
- SMOTE for oversampling
- One-hot encoding for categorical variables
- Log transformation for transaction amounts
- Jupyter Notebook for analysis and visualization
# Key Steps / Methods Used
1.	Data Preparation: Extract temporal features (hour, day, month), encode categorical attributes, log-transform amounts, and impute missing values.
2.	Handling Imbalance: Apply SMOTE and adjust class weights (scale_pos_weight) to address the <1% fraud prevalence.
3.	Modeling: Train and evaluate Random Forest and XGBoost classifiers using 80/20 train-test split and five-fold cross-validation for hyperparameter tuning.
4.	Evaluation: Assess models with precision, recall, F1 score, ROC AUC, and confusion matrices.
# Main Results / Outcomes
- 	Random Forest: Detected 0% of fraud cases due to severe class imbalance, despite high overall accuracy.
- 	XGBoost (scale_pos_weight): Identified 19.5% of fraud cases but generated high false positives, with an overall accuracy of 84.56%.
- 	XGBoost (SMOTE): Maintained high accuracy (97.43%) but failed to meaningfully improve fraud detection (ROC AUC ~0.513).

