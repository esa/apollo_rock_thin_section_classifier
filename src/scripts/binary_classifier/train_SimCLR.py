
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
import tensorflow as tf 
import argparse

from apollo_petro_ai.data_management.process.preprocessing_helper import split_dataset_into_test_and_train_sets
from apollo_petro_ai.machine_learning.network.architecture import SimCLR

learning_rate=0.0001
epochs=150
batch_size=16

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

    training_directory = "0_data/binary/train" # "0_data/group_safe_binary/train"
    validation_directory = "0_data/binary/validation" # "0_data/group_safe_binary/validation"
    all_training = "0_data/binary/all_training" # "0_data/group_safe_binary/all_training"
    test_directory = "0_data/binary/test" # "0_data/group_safe_binary/test"
    
    # best_model = wandb.restore('simclr_model.weights.h5', run_path="freja-thoresen/SimCLR/lrcnvnrs")

    # Change if a users wants to train a different network
    network = SimCLR(path="2_contrastive_learning/models/pretrained_r50_1x_sk0/", learning_rate=learning_rate, momentum=0.9, weight_decay=0., train_dir=training_directory, val_dir=validation_directory, test_dir=test_directory, epochs=epochs, finetune_epochs=5, batch_size=batch_size) #, my_weights=best_model.name)
    
    if args.t:
        network.epochs = 2
        network.finetune_epochs = 2

    if args.c:
        # Do cross validation training
        network.cross_validation_training(args.source, all_training, 5, args.f)
    elif args.x:
        # Run experiment1 and split data into train and test set
        split_dataset_into_test_and_train_sets(args.source, all_training, test_directory, 0.15)
        network.experiment1(args)
    else:
        # Run and evaluate model on validation set
        split_dataset_into_test_and_train_sets(args.source, all_training, test_directory, 0.2)
        network.preprocessing(all_training)
        #network.create_train_val_split_with_group_limit(training_directory, validation_directory, training_directory, validation_directory)
        
        network.training_directory = training_directory
        network.validation_directory = validation_directory
        network.test_directory = test_directory
               
        _, _, model = network.train(args.f)
        # network.evaluate_trained_model(model, validation_directory, args.g)
        # If -T was specified, also evaluate on test data
        if args.T:
            print("--------------- TESTING --------------")
            network.evaluate_trained_model(model, test_directory, args.g)
            
# python train_binary_model.py -f -c -T ../0_data/datasets/rock_type
if __name__ == "__main__":
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="SimCLR",

        # track hyperparameters and run metadata
        config={
        "learning rate": learning_rate,
        "architecture": "SimCLR",
        "dataset": "Breccia/basalt",
        "epochs": epochs,
        "batch size": batch_size,
        "fine tune": False,
        "group safe": False
        }
    )

    main(['/home/spaceship/users/freja_thoresen/thin-slice-classifier/0_data/datasets/rock_type']) # '-c', '-t', -f', '-f',
