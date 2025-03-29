import time
from pprint import pprint
import pickle
import re

import numpy as np
import tensorflow as tf
from keras import applications
from keras import backend as K
from keras import callbacks, layers, losses, optimizers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model, Sequential
from keras.utils import image_dataset_from_directory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA
import wandb
from apollo_petro_ai.data_management.process.preprocessing_helper import (
    generate_splits,
    get_sample_id,
    process_split,
    split_dataset_into_test_and_train_sets,
)
from apollo_petro_ai.machine_learning.network.network_utils import (
    get_class_weights,
    reduce_to_group,
)
from apollo_petro_ai.plotting.create_figures import draw_PR_curve
import os
import shutil
from collections import defaultdict

CROP_PROPORTION = 0.875  # Standard for ImageNet.

analyze_false_positives_data = {"target": None, "predictions": [], "files": []}


class Network(tf.keras.Model):
    def __init__(self, width, height, train_dir, valid_dir, test_dir, **parameters):
        super().__init__()
        self.img_width, self.img_height = width, height
        self.training_directory = train_dir
        self.validation_directory = valid_dir
        self.test_directory = test_dir
        self._epochs = parameters.get("epochs")
        self._finetune_epochs = parameters.get("fine_tune_epochs")
        self.batch_size = parameters.get("batch_size")
        self.learning_rate = parameters.get("learning_rate")
        self.fine_tune_learning_rate = parameters.get("fine_tune_learning_rate")
        self.cut_off = parameters.get("cut_off")
        #self.saved_model, self.model = self.compile_model()
    
    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, ep):
        self._epochs = ep

    @property
    def finetune_epochs(self):
        return self._finetune_epochs

    @finetune_epochs.setter
    def finetune_epochs(self, fine_tune_ep):
        self._finetune_epochs = fine_tune_ep

    def data_augmentation(self):
        data_augmentation = Sequential(name="data_augmentation")
        data_augmentation.add(layers.Rescaling(1.0 / 255))
        data_augmentation.add(layers.RandomFlip(mode="horizontal"))
        data_augmentation.add(layers.RandomFlip(mode="vertical"))
        data_augmentation.add(layers.RandomRotation(0.5)) 
        return data_augmentation       

    def preprocessing(self, images_directory, ratio=0.25):
        """
        Split data from images_directory into a train and validation directory
        Args:
            images_directory: the directory where all the images are stored

        Returns:
            Does not return, but creates training and validation directory with respective images
        """
        split_dataset_into_test_and_train_sets(
            images_directory, self.training_directory, self.validation_directory, ratio
        )

    def pretrained_model(self):
        """
        Get pretrained model, default is VGG16, but can be overriden with other network architectures
        Returns:
            pretrained model
        """
        return applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(self.img_width, self.img_height, 3),
        )

    def evaluate_trained_model(self, model, data_dir, draw_graphs=False):
        """
        Evaluates the trained model
        Args:
            model: trained model
            data_dir: directory where images are stored that we want to model to predict the classes of
            draw_graphs: whether a precision recall curve needs to be drawn

        Returns:
            accuracy when predicting on individual images and accuracy when grouping images based on the same sample ID
            and using collective prediction
        """
        
        # Load evaluation data
        eval_ds = image_dataset_from_directory(
            data_dir,
            image_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            shuffle=False,
            label_mode="binary",
        )

        eval_data = None
        predictions = None
        target_data = None
        
        # Collect all file paths, but only keep label and filename
        eval_ds_files = eval_ds.file_paths
        eval_ds_files = list(map(lambda path: path[path.find("\\") + 1 :], eval_ds_files))
        analyze_false_positives_data["files"] = eval_ds_files

        # Iterate over the dataset, predicting in batches
        for data_batch in eval_ds.as_numpy_iterator():

            def load_data(data, store):
                # Helper function for storing the data correctly
                if store is None:
                    return data
                else:
                    return np.append(store, data, axis=0)

            eval_data = load_data(data_batch[0], eval_data)

            # Use model to predict labels of this batch
            predicted_values = self.model.predict(data_batch[0], verbose=0).flatten()
            predictions = load_data(predicted_values, predictions)
            target_data = load_data(data_batch[1].flatten(), target_data)

        predictions = tf.nn.sigmoid(predictions)
        analyze_false_positives_data["target"] = target_data
        analyze_false_positives_data["predictions"].append(predictions)

        print("---------- Evaluation as Image -----------")
        # Have target_data and predictions be binary. Predictions > 0.5 since we are using sigmoid
        accuracy = accuracy_score(target_data > 0.5, predictions > 0.5)
        print("Accuracy ", accuracy)
        print(classification_report(target_data > 0.5, predictions > 0.5))
        print(confusion_matrix(target_data > 0.5, predictions > 0.5))

        print("----------- Evaluation as Group -----------")
        # Group images based on sample ID and use majority voting to predict label
        groups = list(map(lambda l: get_sample_id(l), eval_ds_files))
        group_predictions = np.asarray(reduce_to_group(predictions, groups))
        group_target = np.asarray(reduce_to_group(target_data, groups))

        # Have target_data and predictions be binary. Predictions > 0.5 since we already only have 0s and 1s
        group_accuracy = accuracy_score(group_target > 0.5, group_predictions > 0.5)
        print("Accuracy ", group_accuracy)
        print(classification_report(group_target > 0.5, group_predictions > 0.5))
        print(confusion_matrix(group_target > 0.5, group_predictions > 0.5))

        if draw_graphs:
            if group_target.shape[1] > 2:
                print("More than 2 categories, cant draw PR curve")
            else:
                draw_PR_curve(group_target[:, 0], group_predictions[:, 0])
        return accuracy, group_accuracy

    def analyze_false_positives(self, args):
        """
        Check for repeated false positives
        Args:
            args: arguments passed from command line

        Returns:
            print of all wrong predictions
        """
        # Get train and validation set
        self.preprocessing(args.source)

        # Train and evaluate model
        _, _ = self.evaluate_trained_model(self.model, self.validation_directory)
        
        t = analyze_false_positives_data["target"]
        # Unpack predictions into flat list
        p = list(map(list, zip(*analyze_false_positives_data["predictions"])))
        # Predictions to binary, denoting class labels
        p = list(map(lambda y: list(map(lambda x: (x[0] > 0) * 1, y)), p))
        f = analyze_false_positives_data["files"]

        # Zip everything together
        data = list(zip(t, f, p))
        # Keep predictions that were incorrect
        data = list(filter(lambda x: sum(x[2]) / len(x[2]) != x[0], data))

        pprint(data)
        
        
    def create_train_val_split_with_group_limit(self, original_train_dir, original_val_dir, new_train_dir, new_val_dir):
        """
        Creates new train and validation directories with at most one image per group.

        Args:
            original_train_dir: Path to the original training directory.
            original_val_dir: Path to the original validation directory.
            new_train_dir: Path to the new training directory to be created.
            new_val_dir: Path to the new validation directory to be created.
        """

        def process_directory(original_dir, new_dir, group_counts):
            for class_name in os.listdir(original_dir):
                class_dir = os.path.join(original_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue  # Skip non-directory entries

                new_class_dir = os.path.join(new_dir, class_name)
                os.makedirs(new_class_dir, exist_ok=True)  # Create class directory if it doesn't exist

                for filename in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, filename)
                    if not os.path.isfile(file_path):
                        continue  # Skip non-file entries

                    group_id = get_sample_id(file_path)
                    if group_counts[group_id] == 0:
                        # Copy the file if it's the first from its group or it's for validation and it's the second
                        shutil.copy2(file_path, new_class_dir)
                        group_counts[group_id] += 1

        # Create new directories if they don't exist
        os.makedirs(new_train_dir, exist_ok=True)
        os.makedirs(new_val_dir, exist_ok=True)

        # Keep track of how many images from each group have been added
        group_counts = defaultdict(int)

        # Process training directory first
        process_directory(original_train_dir, new_train_dir, group_counts)

        # Process validation directory, allowing one more image per group for validation
        process_directory(original_val_dir, new_val_dir, group_counts)
        
    def fetch_for_train(self):
        """
        Used for fetching train and validation data and configuring both datasets for performance.
        Also calculates class weights to accommodate for imbalance in dataset
        Returns:
            training dataset, validation dataset, and class weights
        """

        # Get training dataset
        train_ds = image_dataset_from_directory(
            self.training_directory,
            image_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            label_mode="binary",
        )

        # Get validation dataset
        valid_ds = image_dataset_from_directory(
            self.validation_directory,
            image_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            label_mode="binary",
        )

        # Calculate class weights
        class_weights = get_class_weights(train_ds)

        # Autotune the prefetch buffer size
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
        return train_ds, valid_ds, class_weights

    def fine_tune(self, model, log_path="1_binary_classifier/logs/ft", fold=''):
        """
        Main loop for fine-tuning the model
        Args:
            model: already trained model that is to be fine-tuned
            log_path: where to log information

        Returns:
            History containing information about losses and compiled metrics (in this case accuracy).
            The model after fine-tuning it
        """
        # Get training and validation set as well as class weights
        train_ds, valid_ds, class_weights = self.fetch_for_train()

        # Make sure to save the best model
        callback = callbacks.ModelCheckpoint(
            "1_binary_classifier/output/binary-model-tl{val_accuracy:.2f}.keras",
            save_best_only=True,
        )
        # Early stop if the validation accuracy has not improved in a few epochs, restoring the best weights
        early_stop = callbacks.EarlyStopping(
            monitor="val_accuracy", patience=150, restore_best_weights=True, verbose=1
        )

        # Using SGD this time with a very low learning rate. No need for it to be adaptive
        model.compile(
           loss=losses.BinaryCrossentropy(from_logits=True),
           optimizer=optimizers.SGD(learning_rate=self.fine_tune_learning_rate, momentum=0.9),
           metrics=["accuracy"],
        )
        
        # Train network
        h = model.fit(
            train_ds,
            epochs=self._finetune_epochs,
            validation_data=valid_ds,
            class_weight=class_weights,
            callbacks=[callback, early_stop],
        )
        
        # Log loss to wandb
        # wandb.log({f"Fine-tune loss{fold}": np.array(h.history['loss'])}) 
        # wandb.log({f"Fine-tune val loss{fold}": np.array(h.history['val_loss'])}) 

        # Log loss to wandb for each epoch
        for epoch in range(len(np.array(h.history['loss']))):
            wandb.log({f"Fine-tune loss{fold}": np.array(h.history['loss'][epoch])})
            wandb.log({f"Fine-tune val loss{fold}": np.array(h.history['val_loss'][epoch])})
                
        return h, model

    def compile_model(self):
        # Load and freeze the pretrained transfer learning model
        saved_model = self.pretrained_model()
        saved_model.trainable = False

        # Add some realistic data augmentation to the model
        data_augmentation = self.data_augmentation()

        # Main structure of the model, adding our own classification head at the end
        inputs = Input(shape=(self.img_width, self.img_height, 3))
        features = data_augmentation(inputs)
        features = saved_model(features, training=False)
        features = GlobalAveragePooling2D()(features)
        features = Dense(256, activation="relu")(features)
        features = Dropout(0.2)(features)
        outputs = Dense(1)(features)

        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model. Using Adam as it is an adaptive optimizer
        model.compile(
            loss=losses.BinaryCrossentropy(from_logits=True),
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True),
            metrics=["accuracy"],
        )

        model.summary(
            line_length=None,
            positions=None,
            print_fn=None,
            expand_nested=False,
            show_trainable=False,
            layer_range=None,
        )
        
        return saved_model, model

    def train(self, fine_tune=False, fold=''):
        """
        Main loop for training the network
        Args:
            fine_tune: whether the model is scheduled to be fine-tuned

        Returns:
            The best accuracy, loss and trained model
        """
        # self.saved_model, self.model = self.compile_model()
        
        # Make sure to save the best model
        callback = callbacks.ModelCheckpoint(
            "1_binary_classifier/output/binary-model-tl-{val_accuracy:.2f}.keras",
            save_best_only=True,
        )
        # Early stop if the validation accuracy has not improved in a few epochs, restoring the best weights
        early_stop = callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1
        )

        # Get training and validation set as well as class weights
        train_ds, valid_ds, class_weights = self.fetch_for_train()

        # Train network
        history = self.model.fit(
            train_ds,
            epochs=self._epochs,
            validation_data=valid_ds,
            class_weight=class_weights,
            callbacks=[callback, early_stop],
        )
        
        # Log loss to wandb
        for epoch in range(len(np.array(history.history['loss']))):
            wandb.log({f"Loss{fold}": np.array(history.history['loss'][epoch])}, step=epoch)
            wandb.log({f"Val loss{fold}": np.array(history.history['val_loss'][epoch])}, step=epoch)
             
        # If the model is scheduled to be fine-tuned, unfreeze the transfer learning model and freeze the layers
        # up till the specified layer by cut_off
        if fine_tune:
            self.saved_model.trainable = True
            for layer in self.saved_model.layers[: self.cut_off]:
                layer.trainable = False
            history, model = self.fine_tune(self.model, fold=fold)

        # Since early stopping will restore weights to best performing model, save the accuracy of that model
        acc = max(history.history["val_accuracy"])
        best_epoch = history.history["val_accuracy"].index(acc)
        loss = history.history["val_loss"][best_epoch]
        return acc, loss, model

    def collect_representations(self, layer_id, ds):
        """
        Collect all representations for images from certain layer, perform dimensionality reduction,
        cluster, and calculate metrics
        Args:
            layer_id: the layer to extract representations from

        """
        AUTOTUNE = tf.data.AUTOTUNE

        representations = None
        info_per_sample_id = []

        with open("0_data/combined_data2x.msm", "rb") as f:
            sample_info = pickle.load(f)

        for data_batch in ds.as_numpy_iterator():
            im, _, file_names = data_batch
            # Extract the sample id and save information
            for file in file_names:
                sample_id = re.search(r"((\d+),\d*)", str(file)).group(2)
                info_per_sample_id.append(sample_info.get(sample_id))
            # Collect all flattened representations
            rep = self.get_flattened_representations(im, layer_id)
            if representations is not None:
                representations = np.concatenate((representations, rep), axis=0)
            else:
                representations = rep

        # Apply dimensionality reduction
        n_comp = 2
        pca = PCA(n_components=n_comp)
        vis_ex = pca.fit_transform(representations)

        labels = KMeans(
            init="k-means++", n_clusters=4, n_init=10, max_iter=300, random_state=None
        ).fit_predict(vis_ex)

        print(silhouette_score(vis_ex, labels))

    def get_flattened_representations(self, x, block_group):
        """
        Passes data through model and collects flattened representations
        Args:
            x: current batch of images
            block_group: which block group to extract representations from

        Returns:
            flattened representations from certain block group
        """
        image_data = x
        outputs = self.saved_model(image_data, trainable=False)

        # Flatten and collect outputs of different blocks in the network
        block_out = tf.math.l2_normalize(outputs[block_group], 0)

        block_out = np.reshape(block_out, (block_out.shape[0], -1))

        return block_out

    def visualise_model_representations(self, ds):
        """Handles calls to present results of model"""
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = ds.prefetch(buffer_size=AUTOTUNE)
        for data_batch in train_ds.as_numpy_iterator():
            im, _, _ = data_batch
            self.cluster_representations(im)
            break  # only first batch for now

    def hyperparameter_cross_validation(self, path, splits, fine_tune):
        """
        Train network and do cross validation
        Args:
            path: directory where all images are saved
            splits: how many splits to use for K-fold
            fine_tune: specifying if network should be fine-tuned

        Returns:
            prints fold loss and accuracy as well as final average loss and accuracy over all folds
        """
        
        folds, X, y = generate_splits(path, splits)
        
        scores = []
        losses = []
        group_accuracies = []
        for fold_idx, (train, val) in enumerate(folds):
            print("\nFold: ", fold_idx)
            
            # Move the files into their respective folders, by splitting in validation and train directory
            process_split(X, y, train, val, self.training_directory, self.validation_directory)

            # Reinitialize the model
            self.saved_model, self.model = self.compile_model()
            
            # Train the model
            accuracy, loss, model = self.train(fine_tune)
            accuracy *= 100
            
            # Evaluate on the validation set
            _, group_accuracy = self.evaluate_trained_model(model, self.validation_directory)          
            
            print("Fold loss: %.4f acc: %.4f" % (loss, accuracy))
            losses.append(loss)
            scores.append(accuracy)
            group_accuracies.append(group_accuracy * 100)

            K.clear_session()

        # Print average metrics across all folds
        print(f"Loss: {np.mean(losses):.4f} (+/- {np.std(losses):.4f})")
        print(f"Acc: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
        print(f"Sample accuracy: {np.mean(group_accuracies):.2f} (+/- {np.std(group_accuracies):.2f})")

    def cross_validation(self, path, all_train_path, splits, fine_tune=False):
        """
        Train network and do cross validation
        Args:
            path: directory where all images are saved
            all_train_path: where to save training images temporarily after split of train test
            splits: how many splits to use for K-fold
            fine_tune: specifying if network should be fine-tuned

        Returns:
            prints fold loss and accuracy as well as final average loss and accuracy over all folds
        """
        # Generate folds, ensuring images of the same sample end up in the same class
        folds, X, y = generate_splits(path, splits, True)
        
        scores = []
        losses = []
        test_group_accuracies = []
        test_accuracies = []
        for fold_idx, (train, test) in enumerate(folds):
            print("\nFold: ", fold_idx)
            
            # Move the files into their respective folders, use all_train_path as a temporary directory
            # to split into the validation and train directory later
            process_split(X, y, train, test, all_train_path, self.test_directory)
            
            # Move files from all_train_path to a train and validation directory
            self.preprocessing(all_train_path, ratio=0.4)

            # reset network
            self.saved_model, self.model = self.compile_model()

            # train network
            accuracy, loss, model = self.train(fine_tune, fold=f' {fold_idx}')
            accuracy *= 100

            print("Fold loss: %.4f acc: %.4f" % (loss, accuracy))

            losses.append(loss)
            scores.append(accuracy)

            # Evaluate on the test set
            validation_acc, group_accuracy = self.evaluate_trained_model(model, self.test_directory)
            test_accuracies.append(100 * validation_acc)
            test_group_accuracies.append(100 * group_accuracy)
            
            wandb.log({'Fold loss': loss})
            wandb.log({'Fold accuracy': accuracy})
            wandb.log({'Test accuracy': validation_acc*100})
            wandb.log({'Test group accuracy': group_accuracy*100})
            
            if len(scores) > 1:
                if scores[-1] < min_score:
                    min_score = scores[-1]
                    self.save_model(self.saved_model, f'simclr_binary_finetune_fold_{fold_idx}')
                    self.save_model(self.model, f'simclr_binary_model_fold_{fold_idx}')
            else:
                min_score = scores[0]
                self.save_model(self.saved_model, f'simclr_binary_finetune_fold_{fold_idx}')
                self.save_model(self.model, f'simclr_binary_model_fold_{fold_idx}')
                
            K.clear_session()

        # Print average metrics
        print(f"Loss: {np.mean(losses):.4f} (+/- {np.std(losses):.4f})")
        print(f"Acc: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
        print(f"TEST accuracy: {np.mean(test_accuracies):.2f} (+/- {np.std(test_accuracies):.2f})")
        print(f"TEST group accuracy: {np.mean(test_group_accuracies):.2f} (+/- {np.std(test_group_accuracies):.2f})")

    def save_model(self, model, run_name):
        # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
        model.save_weights(os.path.join(wandb.run.dir, f"{run_name}.weights.h5"), overwrite=True)
        # model.save(os.path.join(wandb.run.dir, f"{run_name}.keras"), overwrite=True)

        # Save a model file manually from the current directory:
        wandb.save(f"{run_name}.weights.h5")
        # wandb.save(f"{run_name}.keras")
        