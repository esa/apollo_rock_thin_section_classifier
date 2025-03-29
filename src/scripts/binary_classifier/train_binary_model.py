import argparse

from apollo_petro_ai.data_management.process.preprocessing_helper import split_dataset_into_test_and_train_sets
from apollo_petro_ai.machine_learning.network.architecture import InceptionResNet
import wandb

def main(args):
    print('in main')
    parser = argparse.ArgumentParser(description='Trains binary classifier on provided dataset')
    parser.add_argument('source', type=str, action="store",
                        help='Source directory for training and validation classes. '
                             'Images should be divided into two folders describing their classes.')
    parser.add_argument('-c', action='store_true', help='Enables cross validation training')
    parser.add_argument('-f', action='store_true', help="Enables fine tuning after training.")
    parser.add_argument('-x', action='store_true', help='Enables experiment 1 - Check for repeated false positives')
    parser.add_argument('-t', action='store_true', help='Runs only 2 epochs - for testing purposes')
    parser.add_argument('-g', action='store_true',
                        help='Draws precision recall curves (not available for cross validation)')
    parser.add_argument('-T', action='store_true', help="Calculate final testing of classifier")
    args = parser.parse_args(args)

    training_directory = "0_data/binary/train"
    validation_directory = "0_data/binary/validation"
    all_training = "0_data/binary/all_training"

    test_directory = "0_data/binary/test"

    best_model = wandb.restore('simclr_model_epoch_28.weights.h5', run_path="freja-thoresen/SimCLR/nozw921x")
    # Change if a users wants to train a different network
    network = InceptionResNet(training_directory, validation_directory, test_directory, learning_rate=0.001, fine_tune_learning_rate=0.0001,
                              epochs=20, fine_tune_epochs=30, batch_size=32, my_weights=best_model.name)
    
    if args.t:
        network.epochs = 2
        network.finetune_epochs = 2

    if args.c:
        # Do cross validation training
        network.cross_validation(args.source, all_training, 10, args.f)
    elif args.x:
        # Run experiment1 and split data into train and test set
        split_dataset_into_test_and_train_sets(args.source, all_training, test_directory, 0.15)
        network.analyze_false_positives(args)
    else:
        # Run and evaluate model on validation set
        split_dataset_into_test_and_train_sets(args.source, all_training, test_directory, 0.15)
        network.preprocessing(all_training)
        _, _, model = network.train(args.f)
        network.evaluate_trained_model(model, validation_directory, args.g)
        # If -T was specified, also evaluate on test data
        if args.T:
            print("--------------- TESTING --------------")
            network.evaluate_trained_model(model, test_directory, args.g)
            
# python train_binary_model.py -f -c -T ../0_data/datasets/rock_type
if __name__ == "__main__":
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Geological Binary Classifier",

        # track hyperparameters and run metadata
        config={
        "learning rate": 0.001,
        "fine tune learning rate": 0.0001,
        "architecture": "SimCLR + binary",
        "dataset": "Breccia/basalt",
        "epochs": 20,
        "batch size": 32,
        "fine tune": True,
        "fine tune epochs": 30,
        "group safe": False,
        "SimCLR model": 'nozw921x'
        }
    )
    
    main(['-f', '-c', '/home/spaceship/users/freja_thoresen/thin-slice-classifier/0_data/datasets/rock_type']) # '-c', '-t', 
