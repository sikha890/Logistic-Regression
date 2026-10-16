# Logistic Regression Classification

## Objective

The objective of this task is to build and evaluate a binary classification model using Logistic Regression.

This task covers:

- Data loading and preprocessing
- Train-test splitting
- Feature standardization
- Logistic Regression model training
- Model evaluation
- Confusion Matrix
- Precision and Recall
- ROC-AUC
- Threshold tuning
- Sigmoid function
- Model observations and interpretation

## Dataset

The Breast Cancer Wisconsin dataset available through `sklearn.datasets` is used for this project.

The dataset contains:

- 569 samples
- 30 numerical features
- Binary target variable

Target classes:

- `0` → Malignant
- `1` → Benign

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Project Workflow

1. Load the Breast Cancer Wisconsin dataset.
2. Convert the dataset into Pandas DataFrames/Series.
3. Check the dataset structure and missing values.
4. Split the data into training and testing sets.
5. Standardize the features using `StandardScaler`.
6. Train a Logistic Regression model.
7. Generate predictions and probabilities.
8. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix
   - ROC-AUC
9. Plot the ROC curve.
10. Tune the classification threshold.
11. Visualize and understand the sigmoid function.
12. Record observations and conclusions.

## Model

Logistic Regression is used as the binary classification algorithm.

The model was trained using:

```python
LogisticRegression(max_iter=1000, random_state=42)
