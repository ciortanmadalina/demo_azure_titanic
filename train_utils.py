import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(train_path, test_path):
    # Load train and test datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    # Combine train and test for preprocessing
    data = pd.concat([train, test], sort=False)

    # Fill missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Drop unnecessary columns
    data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

    # Convert categorical data to numerical data
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])

    # Split data back into train and test
    train = data[:len(train)]
    test = data[len(train):]
    
    return train, test

def prepare_data_for_training(train):
    # Prepare data for training the model
    y_train = train['Survived']
    X_train = train.drop('Survived', axis=1)
    return X_train, y_train

def prepare_test_data(test):
    # Prepare test data (this doesn't include 'Survived' column)
    X_test = test.drop('Survived', axis=1)
    return X_test

def train_model(X_train, y_train):
    # Create a model and train it
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    # Predict the survival for test data
    predictions = model.predict(X_test)
    return predictions


