import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(train_path, test_path):
    """
    Load train and test datasets from CSV files.

    :param train_path: The file path for the training dataset.
    :type train_path: str
    :param test_path: The file path for the test dataset.
    :type test_path: str
    :returns: The training and test datasets.
    :rtype: tuple of pandas.DataFrame
    """
    # Load train and test datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    """
    Preprocess the training and test datasets.
    The preprocessing steps include filling missing values, dropping unnecessary columns,
    and encoding categorical variables.

    :param train: The training dataset.
    :type train: pandas.DataFrame
    :param test: The test dataset.
    :type test: pandas.DataFrame
    :returns: The preprocessed training and test datasets.
    :rtype: tuple of pandas.DataFrame
    """
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
    """
    Prepare the training data for model training.

    :param train: The preprocessed training dataset.
    :type train: pandas.DataFrame
    :returns: The feature matrix and the target array for model training.
    :rtype: tuple of pandas.DataFrame and pandas.Series
    """
    # Prepare data for training the model
    y_train = train['Survived']
    X_train = train.drop('Survived', axis=1)
    return X_train, y_train

def prepare_test_data(test):
    """
    Prepare the test data for making predictions.

    :param test: The preprocessed test dataset.
    :type test: pandas.DataFrame
    :returns: The feature matrix for the test dataset.
    :rtype: pandas.DataFrame
    """
    # Prepare test data (this doesn't include 'Survived' column)
    X_test = test.drop('Survived', axis=1)
    return X_test

def train_model(X_train, y_train):
    """
    Train a Gaussian Naive Bayes model.

    :param X_train: The feature matrix for the training dataset.
    :type X_train: pandas.DataFrame
    :param y_train: The target array for the training dataset.
    :type y_train: pandas.Series
    :returns: The trained model.
    :rtype: sklearn.naive_bayes.GaussianNB
    """
    # Create a model and train it
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    """
    Make predictions on the test data using the```python
trained model.

    :param model: The trained Gaussian Naive Bayes model.
    :type model: sklearn.naive_bayes.GaussianNB
    :param X_test: The feature matrix for the test dataset.
    :type X_test: pandas.DataFrame
    :returns: The predicted outcomes for the test dataset.
    :rtyp
    """
    # Predict the survival for test data
    predictions = model.predict(X_test)
    return predictions


