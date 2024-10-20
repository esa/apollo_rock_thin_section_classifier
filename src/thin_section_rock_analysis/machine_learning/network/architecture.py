import os
from keras import applications

from thin_section_rock_analysis.machine_learning.optimizer.lars_optimizer import LARSOptimizer
from thin_section_rock_analysis.data_management.data_augmentation import preprocess_for_eval, preprocess_for_train

from .Network import Network
from keras.models import load_model
import tensorflow as tf
from keras import callbacks, layers, losses, optimizers
from sklearn.metrics import accuracy_score, silhouette_score
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
from keras_cv.losses import SimCLRLoss
import keras_cv

experiment1_data = {"target": None, "predictions": [], "files": []}


class SimCLR(Network):
    def __init__(self, path, learning_rate, momentum, weight_decay, train_dir, val_dir, test_dir, my_weights="imagenet", **parameters):
        # dimensions of our images.
        img_width, img_height = 299, 299
        self.steps_per_epoch =8
        self.weight_decay=weight_decay
        super().__init__(
            img_width,
            img_height,
            train_dir,
            val_dir,
            test_dir,
            **parameters,
            learning_rate=learning_rate,
            finetune_learning_rate=0.0001,
            cut_off=630,
        )   
        self.my_weights = my_weights
        
        self.sim_clr_loss = SimCLRLoss(temperature=0.5)
        self.contrast_transformer = self.data_augmentation()
        # self.optimizer = LARSOptimizer(
        #     learning_rate=learning_rate,
        #     momentum=momentum,
        #     weight_decay=weight_decay,
        #     exclude_from_weight_decay=[
        #         "batch_normalization",
        #         "bias",
        #         "head_supervised",
        #     ],
        #     name="LARSOptimizer",
        # )
        
        self.model, self.class_head = self.compile_model() #path
    
    def configure_optimizers(self):
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.epochs * self.steps_per_epoch,  # Assuming you have `steps_per_epoch` defined
            alpha=self.learning_rate / 50
        )

        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        return optimizer, lr_scheduler_callback

    def pretrained_model(self):
        return applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False, input_shape=(self.img_width, self.img_height, 3), weights=self.my_weights
        )
    
    def data_augmentation(self):
        contrast_transforms = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            #layers.RandomCrop(height=96, width=96),
            layers.RandomZoom(height_factor=(-0.7, 0.)),  # Randomly zoom in or out            
            layers.Resizing(height=299, width=299),
            layers.RandomBrightness(factor=0.3, value_range=(0, 255)),  # brightness
            layers.RandomContrast(factor=0.3),  # contrast
            keras_cv.layers.RandomSaturation(factor=(0.3, 0.7)),  # saturation
            keras_cv.layers.RandomHue(factor=0.1, value_range=(0, 255)),  # hue
            #layers.Normalization(mean=(0.5), variance=(0.5)),
            layers.Rescaling(1./255),  # Convert to [0, 1] range
        ])
        return contrast_transforms
        
    
    def compile_model(self):
        # Load and freeze the pretrained transfer learning model
        saved_model = self.pretrained_model()
        saved_model.trainable = False

        # Add some realistic data augmentation to the model
        # data_augmentation = self.data_augmentation()

        # Main structure of the model, adding our own classification head at the end
        #inputs = Input(shape=(self.img_width, self.img_height, 3))
        #features = data_augmentation(inputs)
        #features = saved_model(features, training=False)
        #features = GlobalAveragePooling2D()(features)
        #features = Dense(256, activation="relu")(features)
        #features = Dropout(0.2)(features)
        #outputs = Dense(1)(features)

        # Add simple new non-linear classification head
        class_head = Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.Dense(units=128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(units=128),
            ],
            name="head_supervised_new",
        )

        #class_head_model = Model(inputs=inputs, outputs=outputs)
        
        #class_head_model.summary(
        #    line_length=None,
        #    positions=None,
        #    print_fn=None,
        #    expand_nested=False,
        #    show_trainable=False,
        #    layer_range=None,
        #)
        self.optimizer, self.lr_scheduler_callback=self.configure_optimizers()
        self.finetune_optimizer, self.finetune_lr_scheduler_callback=self.configure_optimizers()
        return saved_model, class_head

    def train_step(self, x):
        """
        Forward and backward pass for a single training step.

        Args:
            x: current batch of data

        Returns:
            loss: the computed loss for the batch
        """

        with tf.GradientTape() as tape:
            loss = self.forward(x)  # Compute the loss using your model's forward pass

        # Get trainable weights and calculate gradients
        trainable_variables = self.trainable_variables 
        gradients = tape.gradient(loss, trainable_variables)

        # Apply gradients using the optimizer
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        return loss

    def forward(self, x):
        image_data, label_data = x

        image_aug_1 = self.contrast_transformer(image_data)
        image_aug_2 = self.contrast_transformer(image_data)
        
        outputs_1 = self.class_head(self.model(image_aug_1))
        outputs_2 = self.class_head(self.model(image_aug_2))
        
        loss = self.sim_clr_loss(outputs_1, outputs_2)
        return loss
    
    def train(self, fine_tune=False):
        """Fine-tunes the model for a few epochs with early stopping"""
        self.model.trainable = True
        for layer in self.model.layers[: self.cut_off]:
            layer.trainable = False
        
        # Get training and validation set as well as class weights
        train_ds, valid_ds, class_weights = self.fetch_for_train()
        
        loss_per_epoch = []

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=2,           # Stop if validation loss doesn't improve for 2 consecutive epochs
            #restore_best_weights=True  # Restore model weights to the epoch with the best validation loss
        )

        for epoch in range(self.epochs):
            total_loss = 0 
            num_samples = 0

            for data_batch in train_ds.as_numpy_iterator():
                im, labels = data_batch

                if im.shape[0] != self.batch_size:
                    continue                
                loss = self.train_step((im, labels))
                wandb.log({"Batch loss": loss})
                
                total_loss += loss
                num_samples += im.shape[0]

            average_loss = total_loss / num_samples
            loss_per_epoch.append(average_loss)

            # Evaluate on validation set at the end of each epoch
            val_loss = self.calculate_loss(valid_ds) #valid_ds
            wandb.log({"Loss": average_loss})
            wandb.log({"Validation loss": val_loss})
            wandb.log({"Epoch": epoch})

            print(f'Epoch: {epoch}, loss: {average_loss}, val_loss: {val_loss}')
            self.save_model(self.model, f'simclr_model_epoch_{epoch}')

            # Early stopping check
            #early_stopping.on_epoch_end(epoch, logs={'val_loss': val_loss})
            #if early_stopping.stopped_epoch:
            #    print("Early stopping triggered!")
            #    break
            
            # Update learning rate at the end of each epoch
            # self.lr_scheduler_callback.on_epoch_end(epoch, logs={average_loss.ref()}) 

        self.save_model(self.model, 'simclr_model')
        
        return [], [], self.model

    def save_model(self, model, run_name):
        # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
        model.save_weights(os.path.join(wandb.run.dir, f"{run_name}.weights.h5"), overwrite=True)
        model.save(os.path.join(wandb.run.dir, f"{run_name}.keras"), overwrite=True)

        # Save a model file manually from the current directory:
        wandb.save(f"{run_name}.weights.h5")
        wandb.save(f"{run_name}.keras")

        # Save all files that currently exist containing the substring "ckpt":
        # wandb.save('../logs/*ckpt*')

        # Save any files starting with "checkpoint" as they're written to:
        # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
        return
            
    def fine_tune(self):
        
        """Fine-tunes the model for a few epochs with early stopping"""
        
        # Get training and validation set as well as class weights
        train_ds, valid_ds, class_weights = self.fetch_for_train()
        
        loss_per_epoch = []

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=2,           # Stop if validation loss doesn't improve for 2 consecutive epochs
            #restore_best_weights=True  # Restore model weights to the epoch with the best validation loss
        )

        for epoch in range(self.finetune_epochs):
            total_loss = 0 
            num_samples = 0

            for data_batch in train_ds.as_numpy_iterator():
                im, labels = data_batch
                loss = self.train_step((im, labels))
                wandb.log({"Finetune batch loss": loss})
                
                total_loss += loss
                num_samples += im.shape[0]

            average_loss = total_loss / num_samples
            loss_per_epoch.append(average_loss)
            wandb.log({"Finetune loss": average_loss})

            # Evaluate on validation set at the end of each epoch
            val_loss = self.calculate_loss(valid_ds)
            wandb.log({"Finetune validation loss": val_loss})

            print(f'Epoch: {epoch}, loss: {average_loss}, val_loss: {val_loss}')

            # Early stopping check
            early_stopping.on_epoch_end(epoch, logs={'val_loss': val_loss})
            if early_stopping.stopped_early:
                print("Early stopping triggered!")
                break
            
            # Update learning rate at the end of each epoch
            self.lr_scheduler_callback.on_epoch_end(epoch, logs={average_loss})      
        
        self.model.export("2_contrastive_learning/models/simclr_finetuned.keras")
        
        return

    
    def calculate_loss(self, ds, take=0):
        """Evaluate model performance on dataset"""

        total_loss = 0 
        no_images = 0
        
        if take > 0:
            # Iterate over the validation dataset only once
            for images, labels in ds.take(take): 
                loss = self.forward((images, labels))
                total_loss += loss
                no_images += images.shape[0]
        else:
            # Iterate over the validation dataset only once
            for images, labels in ds.as_numpy_iterator(): 
                loss = self.forward((images, labels))
                total_loss += loss
                no_images += images.shape[0]
            
        return total_loss/no_images

    def calculate_accuracy(self, ds):
        """Evaluate model performance on dataset"""

        all_labels = []
        all_preds = []
        total_loss = 0 
        no_images = 0
        # Iterate over the validation dataset only once
        for images, labels, _ in ds.as_numpy_iterator(): 
            loss, _, logits = self.forward((images, labels))
            total_loss += loss
            logits = logits.numpy()
            all_labels.extend(labels > 0.5)
            all_preds.extend(logits > 0.5)
            no_images += images.shape[0]
        
        print(f'Test accuracy: {accuracy_score(all_labels, all_preds)}')
        print(f'Test loss: {total_loss/no_images}') 
        return accuracy_score(all_labels, all_preds)


    def cluster_representations(self, x):
        """
        Clusters and plots images for user to evaluate performance of the model
        Args:
            x: current batch of images

        """
        image_data = x
        outputs = self.model(image_data, trainable=False)

        # Flatten and collect outputs of different blocks in the network
        block2_out = tf.math.l2_normalize(outputs["block_group2"], 0)
        batch, length, width, channels = block2_out.shape
        visual2 = np.reshape(block2_out, (batch, length * width, channels))

        block3_out = tf.math.l2_normalize(outputs["block_group3"], 0)
        batch, length, width, channels = block3_out.shape
        visual3 = np.reshape(block3_out, (batch, length * width, channels))

        block4_out = tf.math.l2_normalize(outputs["block_group4"], 0)
        batch, length, width, channels = block4_out.shape
        visual4 = np.reshape(block4_out, (batch, length * width, channels))

        # Flatten original image features such that they can potentially be used in algorithms like Kmeans
        flat_im = np.reshape(image_data, (batch, 224 * 224, 3))

        # Help to create figures on the right axes
        def figure_helper(vis_ex, draw_ax, dims):
            for col in range(len(draw_ax)):
                # Perform k-means clustering
                labels = KMeans(
                    init="k-means++",
                    n_clusters=(col + 1) * 2,
                    n_init=10,
                    max_iter=300,
                    random_state=None,
                ).fit_predict(vis_ex)
                labels = np.reshape(
                    labels, (dims[1], dims[2], 1)
                )  # Reshape to non flattened representation
                # Upsample images back to original size (standard is using bilinear interpolation)
                labels = tf.image.resize(labels, [224, 224])

                draw_ax[col].imshow(image_data[index])
                draw_ax[col].imshow(labels, alpha=0.5, cmap="inferno")
                draw_ax[col].tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )

        # Set up layout and start plotting images in batch
        for index in range(image_data.shape[0]):
            fig, ax = plt.subplots(4, 3, figsize=(9, 9))
            fig.tight_layout()
            ax[0, 0].remove()
            ax[0, 1].imshow(image_data[index])
            ax[0, 1].axis("off")
            ax[0, 1].set_title("original image")
            ax[0, 2].remove()
            for i, top_ax in enumerate(ax[1]):
                top_ax.set_title(f"{(i+1)*2} clusters")
            for i, side_ax in enumerate(ax[1:, 0]):
                side_ax.set_ylabel(f"block group{4-i}")

            figure_helper(visual4[index], ax[1, :], block4_out.shape)
            figure_helper(visual3[index], ax[2, :], block3_out.shape)
            figure_helper(visual2[index], ax[3, :], block2_out.shape)

            # Overwrite first row to display k-means clustering on image's original rgb features instead
            # figure_helper(flat_im[index], ax[1, :], image_data.shape)
            # ax[1, 0].set_ylabel('K-means')

        plt.show()


class VGG16(Network):
    def __init__(self, train_dir, val_dir, test_dir, **parameters):
        # dimensions of our images.
        img_width, img_height = 250, 250
        super().__init__(
            img_width,
            img_height,
            train_dir,
            val_dir,
            test_dir,
            **parameters,
            learning_rate=0.0002,
            finetune_learning_rate=0.0001,
            cut_off=15,
        )
        self.saved_model, self.model = self.compile_model()


class VGG19(Network):
    def __init__(self, train_dir, val_dir, test_dir, **parameters):
        # dimensions of our images.
        img_width, img_height = 150, 150
        super().__init__(
            img_width,
            img_height,
            train_dir,
            val_dir,
            test_dir,
            **parameters,
            learning_rate=0.001,
            finetune_learning_rate=0.0001,
            cut_off=17,
        )
        self.saved_model, self.model = self.compile_model()

    def pretrained_model(self):
        return applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(self.img_width, self.img_height, 3),
        )

 
class InceptionResNet(Network):
    def __init__(self, train_dir, val_dir, test_dir, my_weights="imagenet", **parameters):
        # dimensions of our images.
        img_width, img_height = 299, 299
        super().__init__(
            img_width,
            img_height,
            train_dir,
            val_dir,
            test_dir,
            **parameters,
            #learning_rate=0.001,
            #fine_tune_learning_rate=0.0001,
            cut_off=630,
        )
        self.my_weights = my_weights
        self.saved_model, self.model = self.compile_model()

    def pretrained_model(self):
        return applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False, input_shape=(self.img_width, self.img_height, 3), weights=self.my_weights
        )


def performance(network_loc):
    """
    Args:
        network_loc: location the network is saved
    Can be used to check performance of the model
    Returns:

    """
    training_directory = "processing/binary/train"
    validation_directory = "processing/binary/validation"

    test_directory = "processing/binary/test"
    model_test = load_model(network_loc)

    network = InceptionResNet(
        training_directory,
        validation_directory,
        test_directory,
        epochs=0,
        finetune_epochs=0,
        batch_size=16,
    )

    acc, g_acc = network.evaluate_trained_model(model_test, test_directory)
    print("sample accuracy: %.4f group accuracy: %.4f" % (acc, g_acc))
