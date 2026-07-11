from src import generate_split_data, train_from_config, test_model
import argparse  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False, help='Whether to preprocess the data, and generate splits')
    args = parser.parse_args()

    if args.preprocess:
        # Preprocess the data
        generate_split_data()

    # Train the model
    train_from_config() 

    # Evaluate the model
    test_model() 