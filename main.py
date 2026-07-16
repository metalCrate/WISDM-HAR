# Entry-point imports keep the CLI thin and defer all work to the modules.
from src import generate_split_data, train_from_config, test_model
import argparse  

if __name__ == "__main__":
    # Parse CLI flags once so the script can route to the requested pipeline stage.
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False, help='Whether to preprocess the data, and generate splits')
    parser.add_argument('--train', type=bool, default=False, help='Whether to train the model')
    args = parser.parse_args()

    # Run preprocessing explicitly to avoid hidden file generation during normal imports.
    if args.preprocess:
        generate_split_data()

    # Training is opt-in so evaluation can be invoked independently.
    if args.train:
        train_from_config()

    # Always evaluate after setup so the script reports the current model state.
    test_model() 