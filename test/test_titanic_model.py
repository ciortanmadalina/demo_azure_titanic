import sys
import os
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
# pylint: disable=E0401
import train_utils
from train_utils import load_data, preprocess_data, prepare_data_for_training, prepare_test_data, train_model, make_predictions  # replace 'your_script' with the name of your script

class TestTitanicModel(unittest.TestCase):
    """
    A class for unit-testing function in the train_utils.py file
    Args:
        unittest.TestCase this allows the new class to inherit
        from the unittest module
    """

    def test_load_data(self):
        """
        Test case for the `load_data` function.
        Ensure the function returns two pandas DataFrames.
        """
        train, test = load_data('train.csv', 'test.csv')
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)

    def test_preprocess_data(self):
        """
        Test case for the `preprocess_data` function.
        Ensure the function preprocesses the data correctly.
        """
        train, test = load_data('train.csv', 'test.csv')
        train, test = preprocess_data(train, test)
        self.assertFalse(train.isnull().values.any())
        self.assertFalse(test.drop("Survived", axis = 1).isnull().values.any())

    def test_prepare_data_for_training(self):
        """
        Test case for the `prepare_data_for_training` function.
        Ensure the function prepares the data for training correctly.
        """
        train, test = load_data('train.csv', 'test.csv')
        train, test = preprocess_data(train, test)
        X_train, y_train = prepare_data_for_training(train)
        self.assertEqual(X_train.shape[0], y_train.shape[0])

    def test_prepare_test_data(self):
        """
        Test case for the `prepare_test_data` function.
        Ensure the function prepares the test data correctly.
        """
        train, test = load_data('train.csv', 'test.csv')
        train, test = preprocess_data(train, test)
        X_test = prepare_test_data(test)
        self.assertIsInstance(X_test, pd.DataFrame)

    def test_train_model(self):
        """
        Test case for the `train_model` function.
        Ensure the function trains the model correctly.
        """
        train, test = load_data('train.csv', 'test.csv')
        train, test = preprocess_data(train, test)
        X_train, y_train = prepare_data_for_training(train)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_make_predictions(self):
        """
        Test case for the `make_predictions` function.
        Ensure the function makes predictions correctly.
        """
        train, test = load_data('train.csv', 'test.csv')
        train, test = preprocess_data(train, test)
        X_train, y_train = prepare_data_for_training(train)
        X_test = prepare_test_data(test)
        model = train_model(X_train, y_train)
        print("!!!", test.shape, X_test.shape)
        predictions = make_predictions(model, X_test)
        self.assertEqual(len(predictions), len(X_test))

if __name__ == '__main__':
    unittest.main()
