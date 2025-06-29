# Auto Insurance Fraud Prediction Using Machine Learning
# Problem It Solves / Context
Auto insurance fraud imposes substantial costs on insurers and consumers, driving up premiums for all policyholders. As fraud techniques evolve in complexity, manual and rule-based detection systems struggle to keep pace. This project investigates machine learning approaches to identify patterns of fraudulent claims, offering a dynamic, data-driven solution.
# Tools / Technologies Used
-	Python (pandas, scikit-learn, XGBoost)
-	imbalanced-learn (SMOTE)
-	One-hot encoding and ordinal mapping for preprocessing
-	Matplotlib / Seaborn for visualizations
-	Stacking classifier combining logistic regression, Random Forest, and XGBoost
-	Jupyter Notebook for analysis and presentation
# Overview of Data
-	Source: carclaims.csv from Kaggle (Vehicle Insurance Fraud Detection dataset)
-	Records: 15,420 auto insurance claims
-	Fraud Instances: 923 fraudulent claims (~6%)
-	Features (33 total): Timestamp details, accident location, vehicle category, policyholder demographics, claim history, deductible amount, supplements count, and additional behavioral/contextual indicators.
# Key Steps / Methods Used
1.	Data Preprocessing
- Standardized missing-value placeholders ("0", "none") to NaN
- Mapped ordinal categories and binary flags to numeric codes
- Imputed missing categorical values with mode and numeric gaps with zero
- One-hot encoded remaining nominal features
- Performed stratified 80/20 train-test split to maintain fraud ratio
2.	Exploratory Data Analysis
- Stacked bar charts comparing fraud vs. non-fraud across categorical variables
-	Histograms, KDE plots, and box plots for numeric features (vehicle age, past claims, deductibles)
-	Correlation heatmap to assess feature linear relationships
-	ROC and Precision–Recall curves for initial model threshold insights
-	Feature importance ranking using Random Forest
3.	Modeling
-	Logistic Regression: Baseline model for interpretability
-	Random Forest: Addressed imbalance via SMOTE oversampling and class_weight='balanced'
-	XGBoost: Gradient boosting capturing nonlinear feature interactions
-	Stacking Ensemble: Combined logistic regression, Random Forest, and XGBoost with a logistic regression meta-learner
4.	Evaluation
-	Metrics: Precision, recall, F1-score, ROC AUC, and confusion matrix
-	Visualizations: ROC and Precision–Recall curves, feature importance plots
-	Validation: Cross-validation and consistency checks on train-test split
Main Results / Outcomes
-	Logistic Regression: Provided interpretability but limited in handling complex patterns.
-	Baseline Random Forest: High overall accuracy but low recall for fraud cases.
-	SMOTE + Class Weighted Random Forest: Precision 1.00, recall 0.94, F1 0.97; ROC AUC ≈0.99.
-	XGBoost: Precision 0.99, recall 0.96, F1 0.98; ROC AUC 1.00 without synthetic sampling.
-	Stacking Ensemble: Precision 0.99, recall 0.97, F1 0.98; balanced performance combining models.

