# Heart Disease Prediction

The "Heart Disease Prediction" project focuses on predicting the risk of coronary heart disease (CHD) using machine learning models. The dataset used in this project, "framingham.csv," contains various health-related features of individuals, and the target variable is whether the person developed CHD in the next ten years or not.

## Project Overview

1. **Data Loading**: The project loads the heart disease data from the "framingham.csv" file into a pandas DataFrame.

2. **Data Preprocessing**: The dataset is inspected to check for any missing values or data issues. The features and target variable are separated.

3. **Data Splitting**: The data is split into training, validation, and test sets using the `train_test_split` function from scikit-learn.

4. **Decision Tree Model**: An initial decision tree classifier is built using the training data. The model's hyperparameter, `max_depth`, is tuned using the validation set to find the best-performing value.

5. **Model Evaluation**: The accuracy scores of the decision tree model on the training and validation sets are computed to assess its performance.

6. **Over-Sampling**: To address class imbalance in the target variable, Random Over-Sampling is applied to the training data.

7. **Random Forest Model**: A Random Forest classifier is trained on the over-sampled training data. The model's hyperparameters are tuned using cross-validation and a grid search.

8. **Model Evaluation**: The accuracy scores of the Random Forest model on the training and test sets are calculated to evaluate its performance.

9. **Gradient Boosting Model**: A Gradient Boosting classifier is trained on the over-sampled training data. Hyperparameters are tuned using cross-validation and a grid search.

10. **Model Evaluation**: The accuracy scores of the Gradient Boosting model on the training and test sets are computed to assess its performance.

11. **Feature Importance**: The feature importances from the Random Forest model are extracted and visualized to identify the most important features in predicting CHD.

12. **Confusion Matrix**: A Confusion Matrix is generated for the best-performing model (Random Forest) to visualize the model's performance on the test set.

## Dependencies

This project requires the following Python libraries:

- matplotlib
- numpy
- pandas
- scikit-learn
- imblearn
- seaborn
- category_encoders
- teaching_tools
  
## Usage

1. Clone this repository to your local machine.
2. Place the "framingham.csv" file in the same directory as the code files.
3. Run the provided Python script to execute the heart disease prediction models.
4. The script will display the accuracy scores of the Decision Tree, Random Forest, and Gradient Boosting models, along with the feature importance plot and the Confusion Matrix for the best-performing model.

## Conclusion

The "Heart Disease Prediction" project demonstrates the application of decision tree-based models for predicting heart disease risk. By evaluating multiple models and optimizing hyperparameters, we aim to identify the most effective model for this particular classification task. The feature importance analysis further provides insights into the significant predictors of coronary heart disease.
