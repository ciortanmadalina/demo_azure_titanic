import train_utils

# Using the functions:
train_path = 'train.csv'
test_path = 'test.csv'

train, test = train_utils.load_data(train_path, test_path)
train, test = train_utils.preprocess_data(train, test)
X_train, y_train = train_utils.prepare_data_for_training(train)
X_test = train_utils.prepare_test_data(test)
model = train_utils.train_model(X_train, y_train)
predictions = train_utils.make_predictions(model, X_test)

print("Done")