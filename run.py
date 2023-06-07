import train_utils
import argparse
# Using the functions:
train_path = 'data/train.csv'
test_path = 'data/test.csv'

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered_model_name", type=str, default="my_model", help="model name")
    args = parser.parse_args()

    registered_model_name = args.registered_model_name
    
    train, test = train_utils.load_data(train_path, test_path)
    train, test = train_utils.preprocess_data(train, test)
    X_train, y_train = train_utils.prepare_data_for_training(train)
    X_test = train_utils.prepare_test_data(test)
    model = train_utils.train_model(X_train, y_train, registered_model_name)
    predictions = train_utils.make_predictions(model, X_test)


if __name__ == "__main__":
    main()  
    print("Done")