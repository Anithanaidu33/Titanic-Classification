```python
pip install numpy pandas scikit-learn

```

    Requirement already satisfied: numpy in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (1.24.3)
    Requirement already satisfied: pandas in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (2.0.3)
    Requirement already satisfied: scikit-learn in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (1.3.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from pandas) (2023.3)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from pandas) (2023.3)
    Requirement already satisfied: scipy>=1.5.0 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from scikit-learn) (1.11.1)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from scikit-learn) (1.3.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from scikit-learn) (3.2.0)
    Requirement already satisfied: six>=1.5 in c:\users\anitha\appdata\local\programs\python\python311\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip is available: 23.2 -> 23.2.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)

# Drop columns if they exist in the DataFrame
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']
df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

# Fill missing values in the 'Age' column with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Separate features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

```

    Accuracy: 0.7640449438202247
    Confusion Matrix:
    [[91 20]
     [22 45]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.81      0.82      0.81       111
               1       0.69      0.67      0.68        67
    
        accuracy                           0.76       178
       macro avg       0.75      0.75      0.75       178
    weighted avg       0.76      0.76      0.76       178
    
    
