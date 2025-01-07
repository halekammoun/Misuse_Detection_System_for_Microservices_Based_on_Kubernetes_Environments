# Misuse_Detection_System_for_Microservices_Based_on_Kubernetes_Environments

This project demonstrates the process of misuse detection in container environments using a dataset obtained from Kaggle. The main steps include downloading the dataset, data preprocessing, training a Random Forest classifier, and evaluating its performance.

## Dataset

The dataset used in this project is the [Misuse Detection in Containers Dataset](https://www.kaggle.com/yigitsever/misuse-detection-in-containers-dataset). It contains labeled data indicating whether container misuse has occurred.

## Project Workflow

### 1. Dataset Download

The dataset is downloaded from Kaggle using the following command:
```bash
!curl -L -o misuse-detection-in-containers-dataset.zip https://www.kaggle.com/api/v1/datasets/download/yigitsever/misuse-detection-in-containers-dataset
```
After downloading, it is extracted using:
```bash
!unzip misuse-detection-in-containers-dataset.zip
```

### 2. Data Preprocessing

```python
import numpy as np
import pandas as pd

# Load the dataset
data_file_path = 'dataset.csv'
df = pd.read_csv(data_file_path)

# Replace inf values with NaN and drop NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Drop irrelevant columns
df = df.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1)

# Define features (X) and target (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### 3. Model Training

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
clf.fit(X_train, y_train)
```

### 4. Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=list(map(str, range(12))))  # Adjust target_names as needed
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```

### 5. Model Persistence

```python
import joblib

# Save the model
joblib.dump(clf, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")

# Load the model
loaded_model = joblib.load('random_forest_model.pkl')

# Use the loaded model for predictions
y_pred_loaded = loaded_model.predict(X_test)
print("Predictions:", y_pred_loaded)
```

## Usage

### Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Joblib

### Steps to Run

1. Clone the repository.
2. Download the dataset using the provided command.
3. Run the provided script to preprocess data, train the model, and evaluate its performance.
4. Use the saved model for predictions on new data.

## Results

- The model achieved an accuracy of approximately 98%.
- The classification report provides detailed precision, recall, and F1-score metrics for each class.

## Conclusion

This project illustrates the end-to-end process of building a misuse detection system for container environments using machine learning.


