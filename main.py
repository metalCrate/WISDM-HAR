from src import generate_split_data, train_from_config, test_model
import argparse  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False, help='Whether to preprocess the data, and generate splits')
    parser.add_argument('--train', type=bool, default=False, help='Whether to train the model')
    args = parser.parse_args()

    if args.preprocess:
        # Preprocess the data
        generate_split_data()

    if args.train:
        # Train the model
        train_from_config()

    # Evaluate the model
    test_model() 