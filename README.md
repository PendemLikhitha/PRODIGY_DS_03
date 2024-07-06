# ProdigyInfoTech_TASK3
## TASK 3: Decision Tree Classifier for Customer Purchase Prediction

This project demonstrates how to build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data from the Bank Marketing dataset.

## Dataset

The dataset used in this project is the Bank Marketing dataset, available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

## Steps

1. **Data Loading**: Load the dataset and inspect columns.
2. **Data Preprocessing**: Encode categorical variables and split data into training and testing sets.
3. **Model Building**: Build a decision tree classifier with specified pruning parameters.
4. **Model Evaluation**: Evaluate the classifier using accuracy score and classification report.
5. **Visualization**: Visualize the decision tree to understand its structure.

## How to Run

1. Clone the repository.
2. Ensure you have the necessary libraries installed (`pandas`, `scikit-learn`, `matplotlib`).
3. Place the dataset (`bank.csv`) in the same directory as the script.
4. Run the script (`decision_tree_classifier.py`) to train the model and generate visualizations.

## Visual Outputs

### Decision Tree Visualization

![Screenshot 2024-07-03 220337](https://github.com/PendemLikhitha/ProdigyInfoTech_TASK3/assets/159911587/a82182f5-612a-400c-bc5c-8ce88ba13880)



## Code

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset with the correct delimiter
df = pd.read_csv('C:\\Users\\91812\\Pictures\\Bank.csv', sep=';')

# Print columns to inspect their names and contents
print("Columns in the dataset:")
print(df.columns)
```

![Screenshot 2024-07-06 221213](https://github.com/PendemLikhitha/PRODIGY_DS_03/assets/159911587/10a3e8a4-b4ee-429e-b3a8-0eba53f6355e)

```python

# Assuming 'y' is the target variable indicating purchase (yes/no)
X = df.drop(columns=['y'])  # Features
y = df['y']                 # Target variable

# Encode categorical variables
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
print(X_encoded )

```
![Screenshot 2024-07-06 221239](https://github.com/PendemLikhitha/PRODIGY_DS_03/assets/159911587/95a4a434-c675-459b-8fc7-9a4a33896b94)
![Screenshot 2024-07-06 221228](https://github.com/PendemLikhitha/PRODIGY_DS_03/assets/159911587/7d9b6888-65ab-40ab-9e52-cb06e7adec8e)

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize decision tree classifier with pruning parameters
clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=20, min_samples_leaf=10)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

```
![Screenshot 2024-07-06 221246](https://github.com/PendemLikhitha/PRODIGY_DS_03/assets/159911587/f4d0e794-39e4-401b-b191-12a0eea125af)

```python
# Visualize the decision tree
plt.figure(figsize=(20, 10))  # Set the figure size
plot_tree(clf, filled=True, feature_names=list(X_encoded.columns), rounded=True, fontsize=12, class_names=['No Purchase', 'Purchase'])
plt.show(block=True)
```

![Screenshot 2024-07-06 221312](https://github.com/PendemLikhitha/PRODIGY_DS_03/assets/159911587/235a9035-61c8-49b3-a353-dd721bf9a8c8)



