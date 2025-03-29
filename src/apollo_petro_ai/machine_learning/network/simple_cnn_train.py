import argparse
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K

from keras import optimizers
from keras import callbacks

from keras.utils import image_dataset_from_directory

from apollo_petro_ai.data_management.process.preprocessing_helper import process_split, generate_splits, split_dataset_into_test_and_train_sets
import time
import numpy as np

training_directory = "processing/binary/train"
test_directory = "processing/binary/test"
# dimensions of our images.
img_width, img_height = 150, 150

epochs = 25
batch_size = 32
n_folds = 5


def create_model():
    """
    Creates, compiles and plots neural network
    Returns:
        Compiled model
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    inputs = layers.Input(shape=input_shape)

    data_augmentation = Sequential(name='data_augmentation')
    data_augmentation.add(layers.Rescaling(1. / 255))
    data_augmentation.add(layers.RandomFlip(mode='horizontal'))
    data_augmentation.add(layers.RandomFlip(mode='vertical'))
    data_augmentation.add(layers.RandomZoom(0.2))

    features = data_augmentation(inputs)
    conv1 = Conv2D(32, 3, activation='relu')(features)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = Flatten()(pool3)
    flat_drop = Dropout(0.5)(flat)
    d1 = Dense(64, activation='relu')(flat_drop)
    d1_drop = Dropout(0.5)(d1)
    last = Dense(1, activation='sigmoid')(d1_drop)
    model = Model(inputs=inputs, outputs=last)
    
    optimizer = optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def train(start, run_id):
    """
    Main loop for training the network
    Args:
        start: string indicating start time (used for logging)
        run_id: indicator to keep track of network versions

    Returns:
        Validation loss and accuracy.
        Saves trained model
    """

    model = create_model()

    train_ds = image_dataset_from_directory(
        training_directory,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary')

    valid_ds = image_dataset_from_directory(
        test_directory,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary')

    train_ds = train_ds.prefetch(buffer_size=10)
    valid_ds = valid_ds.prefetch(buffer_size=10)

    current_time = str(run_id)
    callback = callbacks.ModelCheckpoint("output/binary-model_{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5",
                                         save_best_only=True)
    callback2 = callbacks.TensorBoard(log_dir="logs/" + start + "/" + current_time)
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=[callback, callback2, early_stop])

    model.save('first_model_' + str(run_id) + ".h5")
    # Since early stopping will restore weights to best performing model, save the accuracy of that model
    acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(acc)
    loss = history.history['val_loss'][best_epoch]
    return acc, loss


def cross_valid_train(path, n_splits):
    """
    Performs K-fold cross validation
    Args:
        path: location where labeled images are stored
        n_splits: how many splits for the K-fold cross validation

    Returns:
        Does not return, but prints statistics of the model's performance
    """
    folds, X, y = generate_splits(path, n_splits)
    losses = []
    scores = []
    start = time.strftime("%Y%m%d-%H%M%S")
    for j, (train, val) in enumerate(folds):
        print('\nFold: ', j)
        process_split(X, y, train, val, training_directory, test_directory)
        acc, loss = train(start, j)

        print("Fold loss: %.4f acc: %.4f " % (loss, acc))
        losses.append(loss)
        scores.append(acc)

    print("Loss: {:.4f} (+/- {:.4f})".format(np.mean(losses), np.std(losses)))
    print("Acc: {:.2f} (+/- {:.2f})".format(np.mean(scores), np.std(scores)))


def single_run(path):
    """
    Train model once
    Args:
        path: location where labeled images are stored

    Returns:
        Does not return, but prints statistics of the model's performance
    """
    split_dataset_into_test_and_train_sets(path, training_directory, test_directory, 0.25)
    start = time.strftime("%Y%m%d-%H%M%S")
    acc, loss = train(start, 0)
    print("loss: %.4f acc: %.4f " % (loss, acc))


def main():
    parser = argparse.ArgumentParser(description='Trains binary classifier on provided dataset')
    parser.add_argument('source', type=str, action="store",
                        help='Source directory for training and validation classes. '
                             'Images should be divided into two folders describing their classes.')

    args = parser.parse_args()
    # single_run(args.source)
    cross_valid_train(args.source, n_folds)


if __name__ == '__main__':
    main()
