# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading the data...")
df = pd.read_csv('train.csv') 

# Explore the data
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data Preprocessing
print("\nCleaning the data...")

# Handle missing values
# For numerical columns, fill with median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].median(), inplace=True)

# For categorical columns, fill with mode (most frequent value)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

# Verify no missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Convert categorical text data to numerical values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Display the first 5 rows again to see the changes
print("\nFirst 5 rows after encoding:")
print(df.head())

# Prepare the data for trainin
print("\nPreparing data for training...")

# Drop the Loan_ID column as it's not useful for prediction
df = df.drop('Loan_ID', axis=1)

# Separate Features (X) and Target (y)
X = df.drop('Loan_Status', axis=1)  # Everything except Loan_Status
y = df['Loan_Status']               # Only the Loan_Status column

# Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shape of Training Data:", X_train.shape)
print("Shape of Testing Data:", X_test.shape)

# Train Models
print("\nTraining the machine learning models...")

# Initialize the models
logistic_model = LogisticRegression(max_iter=200, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the models on the training data
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_logistic = logistic_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate Models
print("\nEvaluating model performance...")

print("\n" + "="*50)
print("LOGISTIC REGRESSION PERFORMANCE")
print("="*50)
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))

print("\n" + "="*50)
print("RANDOM FOREST PERFORMANCE")
print("="*50)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:") # This gives Precision, Recall, F1-Score!
print(classification_report(y_test, y_pred_rf))

# Generate a confusion matrix for the Random Forest model (usually the better one)
print("\nGenerating confusion matrix plot...")
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('confusion_matrix.png') # Save the plot for your report
print("Plot saved as 'confusion_matrix.png'")
# plt.show() # You can uncomment this if you want to see it pop up