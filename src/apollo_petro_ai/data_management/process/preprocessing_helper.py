import os
import random
import shutil
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ...machine_learning.network.stratified_group_kfold import StratifiedGroupKFold


def copy_categories(X, y, destination_dir):
    """Copy files from their original directory to the newly constructed train or test directory
    specified by destination_dir
    Args:
        X: paths to images
        y: labels specifying class
        destination_dir: what base directory the file should be moved to (likely train or test)

    Returns:
        Does not return but creates new a new directory where all images specified in X are moved to
    """
    for path, label in zip(X, y):
        # Construct paths where to place the current image being processed
        category_dir = destination_dir + "/" + label
        if not os.path.exists(category_dir):
            os.mkdir(category_dir)
        _, filename = os.path.split(path)
        target = category_dir + "/" + filename
        # Copy from original location to destination
        shutil.copy(path, target)


def process_split(X, y, t_fold, test_fold, training_dir, testing_dir):
    """Helper function to retrieve the images and labels for this particular fold and move the respective images and labels
    in the training_dir and testing_dir accordingly.

    Args:
        X: paths of all images
        y: labels of images specifying the class
        t_fold: all indices to be taken for the training set
        test_fold: all indices to be taken for the testing set
        training_dir: directory where images for training are going to end up
        testing_dir: directory where images for testing are going to end up

    Returns:
        Does not return but helps to create new testing and training directories
        where all images specified in X are moved to according to the way X and y should be sliced into the respective
        training and testing parts.
    """
    print("preprocessing splits")

    # Somehow faster than numpy slicing even though it looks dumb
    X_train = [X[i] for i in t_fold]
    y_train = [y[i] for i in t_fold]

    X_test = [X[i] for i in test_fold]
    y_test = [y[i] for i in test_fold]

    if training_dir.count("/") > 1:
        print("t: ", training_dir)
        shutil.rmtree(training_dir, ignore_errors=True)
        os.makedirs(training_dir)
    else:
        print("Refusing!")

    if testing_dir.count("/") > 1:
        shutil.rmtree(testing_dir, ignore_errors=True)
        os.makedirs(testing_dir)
    else:
        print("Refusing!")

    copy_categories(X_train, y_train, training_dir)
    copy_categories(X_test, y_test, testing_dir)


def get_sample_id(filename):
    return filename.split(",")[0]


def generate_splits(all_data_dir, n_splits, preserve_groups=True):
    """Assumes the classes are split into directories after which the images are collected along with the folder name
    of that specific class directory as a label. Then creates n_splits=K folds to use for later
    Args:
        all_data_dir: directory where all images are saved
        n_splits: essentially specifying the K in K-folds i.e. how many splits to make
        preserve_groups: whether to group photos based on sample id

    Returns:
        The created folds, all image paths and their respective labels
    """
    paths = []
    labels = []
    groups = []
    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)
        # Avoid adding the root, all_data_dir, as a category
        if category_name != os.path.basename(all_data_dir):
            print("cat: ", category_name)
            for file in files:
                # Save paths and directory name as label
                input_file = os.path.join(subdir, file)
                paths.append(input_file)
                labels.append(category_name)
                groups.append(get_sample_id(file))
    # We do not want the files to be sorted based on class so shuffle the paths, labels, and groups in the same way
    temp = list(zip(paths, labels, groups))
    random.shuffle(temp)
    if len(temp) == 0:
        raise ValueError(
            "The data folder is empty or does not exist, therefore length of files is zero."
        )
    paths, labels, groups = map(list, zip(*temp))
    if preserve_groups:
        k_fold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
        # k_fold = GroupShuffleSplit(n_splits=n_splits)
        folds, paths, labels = k_fold.split(paths, labels, groups)
    else:
        # Use K-fold to create splits, uses ~1/n_splits% for testing
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        folds = k_fold.split(paths, labels, group=None)
    return folds, paths, labels


def split_dataset_into_test_and_train_sets(
    all_data_dir,
    training_data_dir,
    testing_data_dir,
    testing_data_pct,
    preserve_groups=True,
    seed=None,
):
    """Helper to split the dataset into a train and test set whilst keeping the directory indicating the class intact.

    Args:
        all_data_dir: where all data is saved
        training_data_dir: the folder where the training data should be copied to
        testing_data_dir: the folder where the testing data should be copied to
        testing_data_pct: the percentage of samples that should be in the test dataset
        preserve_groups: whether to make sure images of same sample end up in same folder
        seed: for reproducibility purposes, can seed np random
    Returns:
        Does not return but creates respective train and test directories.
    """
    # seed random state
    np.random.seed(seed)

    # Recreate testing and training directories
    if testing_data_dir.count("/") > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=True)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print(
            "Refusing to delete testing data directory "
            + testing_data_dir
            + " as we prevent you from doing stupid things!"
        )

    if training_data_dir.count("/") > 1:
        shutil.rmtree(training_data_dir, ignore_errors=True)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print(
            "Refusing to delete testing data directory "
            + training_data_dir
            + " as we prevent you from doing stupid things!"
        )

    num_training_files = 0
    num_testing_files = 0

    sample_dataset_map: Dict[str, Any] = {}
    # Create folders in training and testing directory based on folder names in all_data_dir
    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + "/" + category_name
        testing_data_category_dir = testing_data_dir + "/" + category_name
        print(training_data_category_dir)
        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            sample = get_sample_id(file)
            # If groups are to be preserved, make sure files belonging to the same sample end up in the same directory
            if preserve_groups:
                if sample in sample_dataset_map:
                    belongs_to_testing = sample_dataset_map[sample]
                else:
                    belongs_to_testing = np.random.rand(1) < testing_data_pct
                    sample_dataset_map[sample] = belongs_to_testing
            else:
                belongs_to_testing = np.random.rand(1) < testing_data_pct

            if belongs_to_testing:
                shutil.copy(
                    input_file, testing_data_dir + "/" + category_name + "/" + file
                )
                num_testing_files += 1
            else:
                shutil.copy(
                    input_file, training_data_dir + "/" + category_name + "/" + file
                )
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")


def test():
    folds, X, y = generate_splits("../datasets/rock_type", 5)
    for train, val in folds:
        process_split(
            X, y, train, val, "processing/binary/train", "processing/binary/test"
        )


if __name__ == "__main__":
    test()
