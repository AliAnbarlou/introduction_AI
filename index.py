import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Convert string data to numbers
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'S': 1, 'Q': 2})
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'S': 1, 'Q': 2})

# Fill missing values
imputer = SimpleImputer(strategy='mean')
train_data[['Age', 'Fare']] = imputer.fit_transform(train_data[['Age', 'Fare']])
test_data[['Age', 'Fare']] = imputer.transform(test_data[['Age', 'Fare']])

# Define features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_data['Survived']

# Train the model 5 times and get the average accuracy
avg_accuracy = 0
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = RandomForestClassifier(random_state=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    avg_accuracy += accuracy / 5

print("Average accuracy:", avg_accuracy)

# Plot model predictions and y_tests
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = RandomForestClassifier(random_state=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    ax = axs[i // 3, i % 3]
    ax.scatter(y_test, y_pred, label='Predicted', alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfectly Predicted')
    ax.set_title(f'Model {i+1} - Accuracy: {accuracy:.2f}')
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
    ax.legend()

plt.tight_layout()
plt.show()
