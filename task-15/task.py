import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the Adult Income dataset
from sklearn.datasets import fetch_openml
adult = fetch_openml('adult', version=2, as_frame=True)
df = adult.frame

# Data Preprocessing
df = df.dropna()
label_encoder = LabelEncoder()

# Encode categorical features
for col in df.select_dtypes(include='category').columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop(columns='class')
y = label_encoder.fit_transform(df['class'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (especially important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Applying Cross-Validation to Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# k-fold cross-validation
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
rf_cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')
print(f'Random Forest Cross-Validation Scores: {rf_cv_scores}')
print(f'Mean Cross-Validation Accuracy: {rf_cv_scores.mean()}')

# 2. Investigating Overfitting and Underfitting in Gradient Boosting Machines
train_accuracies = []
val_accuracies = []
n_estimators = [50, 100, 200]
learning_rates = [0.01, 0.1, 0.5]

for lr in learning_rates:
    for n in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        
        # Training accuracy
        train_acc = accuracy_score(y_train, gb.predict(X_train))
        train_accuracies.append(train_acc)
        
        # Validation accuracy
        val_acc = accuracy_score(y_test, gb.predict(X_test))
        val_accuracies.append(val_acc)

# Plotting training and validation accuracies
fig, ax = plt.subplots()
ax.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
ax.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
ax.set_xlabel('Model Configurations (Learning Rates and Estimators)')
ax.set_ylabel('Accuracy')
ax.legend()
plt.title('Gradient Boosting: Training vs Validation Accuracy')
plt.show()

# 3. Evaluating Precision, Recall, and F1-Score for Random Forests
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Precision, Recall, and F1-Score for Random Forest
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f'Random Forest - Precision: {precision_rf}')
print(f'Random Forest - Recall: {recall_rf}')
print(f'Random Forest - F1-Score: {f1_rf}')

# 4. ROC Curve and AUC for Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_probs_gb = gb.predict_proba(X_test)[:, 1]

# ROC curve
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_probs_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Plot ROC curve
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting Classifier')
plt.legend(loc='lower right')
plt.show()

print(f'AUC for Gradient Boosting: {roc_auc_gb}')

# 5. Model Performance Comparison with Different Metrics
# SVM Classifier
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Gradient Boosting already fitted above
y_pred_gb = gb.predict(X_test)

# Evaluate models using cross-validation (Accuracy, Precision, Recall, F1-Score, AUC)
models = {'Random Forest': rf, 'SVM': svm, 'Gradient Boosting': gb}
metrics = {}

for model_name, model in models.items():
    acc = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    if model_name != 'SVM':
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)

    metrics[model_name] = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc_score}

# Print performance comparison
for model_name, metric in metrics.items():
    print(f'\n{model_name} Performance:')
    for key, value in metric.items():
        print(f'{key}: {value:.4f}')
       