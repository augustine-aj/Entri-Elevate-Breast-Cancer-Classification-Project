# Breast-Cancer-Classification-Project

## Overview

This project focuses on classifying breast cancer data using various machine learning algorithms. The primary objective is to compare the performance of different classification models on the Breast Cancer Wisconsin dataset.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin dataset from `scikit-learn`. It includes features computed from breast cancer cell nuclei and labels indicating whether the cancer is malignant or benign.

## Preprocessing Steps

1. **Loading the Dataset**: The dataset is loaded using `scikit-learn`'s `load_breast_cancer` function.

2. **Converting to DataFrame**: The dataset is converted into a `pandas` DataFrame for easier manipulation and analysis.

3. **Mapping Target Values**: Target values (0 and 1) are mapped to descriptive class names ('malignant' and 'benign') for better interpretability.

4. **Handling Missing Values**: The dataset does not contain missing values. If it did, handling missing values would be crucial for maintaining model performance.

5. **Feature Scaling**: Standard scaling is applied to normalize the features, ensuring each feature contributes equally to the model and improving convergence.

## Classification Algorithms

### 1. Logistic Regression

**Description**: Logistic Regression is a linear model used for binary classification tasks. It estimates probabilities using the logistic function.

**Suitability**: Effective for cases where the relationship between the features and the target is approximately linear.

### 2. Decision Tree Classifier

**Description**: Decision Trees split the data into subsets based on the most significant feature at each step, represented as a tree structure.

**Suitability**: Suitable for handling non-linear relationships and interactions between features.

### 3. Random Forest Classifier

**Description**: Random Forest is an ensemble method that combines multiple decision trees to enhance prediction accuracy and control overfitting.

**Suitability**: Effective for large datasets and complex relationships between features.

### 4. Support Vector Machine (SVM)

**Description**: SVM identifies the hyperplane that best separates the classes in the feature space. It is effective in high-dimensional spaces.

**Suitability**: Useful for non-linearly separable data with a clear margin of separation.

### 5. k-Nearest Neighbors (k-NN)

**Description**: k-NN classifies data points based on the majority vote of their k nearest neighbors.

**Suitability**: Works well with small to medium-sized datasets and when decision boundaries are not linear.

## Performance Comparison

The performance of each algorithm is evaluated based on accuracy, precision, recall, and F1 score. A confusion matrix heatmap is also generated for visual comparison.

- **Best Performing Model**: Logistic Regression with Accuracy: 0.9737
- **Worst Performing Model**: 

## Conclusion

This project illustrates the application of various classification algorithms to the Breast Cancer Wisconsin dataset. It provides insights into their effectiveness and helps in understanding which models perform best for this type of data.

