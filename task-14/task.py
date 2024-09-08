
import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load the Titanic dataset
from sklearn.datasets import fetch_openml
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame

# Data Preprocessing
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = df.dropna(subset=['age', 'fare', 'embarked'])
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())

X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
y = df['survived']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Logistic Regression with Cross-Validation
log_reg = LogisticRegression(max_iter=500)

# Single train-test split evaluation
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with single train-test split: {accuracy}')

# Cross-validation evaluation
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(log_reg, X, y, cv=kfold, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')

# 2. Analyzing Overfitting and Underfitting in Decision Trees
train_accuracies = []
val_accuracies = []
max_depths = range(1, 20)

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    # Training accuracy
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    train_accuracies.append(train_acc)
    
    # Validation accuracy
    val_acc = accuracy_score(y_test, tree.predict(X_test))
    val_accuracies.append(val_acc)

# Plotting the training and validation accuracies
plt.plot(max_depths, train_accuracies, label='Training Accuracy')
plt.plot(max_depths, val_accuracies, label='Validation Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree: Training and Validation Accuracy')
plt.show()


# Convert target labels to integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

# 3. Precision, Recall, and F1-Score for Logistic Regression
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}') 


# 4. ROC Curve Analysis for Decision Trees
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Get prediction probabilities
y_probs = tree.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()

print(f'AUC: {roc_auc}')

# 5. Comparing Model Performance with and without Cross-Validation
log_reg_cv_scores = cross_val_score(log_reg, X, y, cv=kfold, scoring='accuracy')
log_reg_mean_cv = log_reg_cv_scores.mean()

tree_cv_scores = cross_val_score(tree, X, y, cv=kfold, scoring='accuracy')
tree_mean_cv = tree_cv_scores.mean()

print(f'Logistic Regression Accuracy (CV): {log_reg_mean_cv}')
print(f'Decision Tree Accuracy (CV): {tree_mean_cv}')

log_reg_acc = accuracy_score(y_test, y_pred)
tree_acc = accuracy_score(y_test, tree.predict(X_test))

print(f'Logistic Regression Accuracy (Train-Test Split): {log_reg_acc}')
print(f'Decision Tree Accuracy (Train-Test Split): {tree_acc}')
