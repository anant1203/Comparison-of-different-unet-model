import gc
import numpy as np
import tensorflow as tf
import argparse
import random
from model.unet_v3 import unet_v3
from model.unet import unet_model
from model.mobile_net import mobilenet_model
from model.attention_unet import attention_unet

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Load training data
def load_data(image_path, mask_path):
    train_images = np.load(image_path)
    train_masks = np.load(mask_path)
    # train_masks = train_masks[..., np.newaxis]
    print("**********")
    print(f"Train images shape: {train_images.shape}, Train masks shape: {train_masks.shape}")
    return train_images, train_masks

# Define dynamic learning rate function
def dynamic_lr(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.5  # Reduce learning rate every 10 epochs
    return lr

# Custom callback to adjust momentum dynamically by recreating the optimizer
class DynamicMomentum(tf.keras.callbacks.Callback):
    def __init__(self, model, momentum_schedule):
        super(DynamicMomentum, self).__init__()
        self.model = model
        self.momentum_schedule = momentum_schedule

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate the new momentum value
        new_momentum = self.momentum_schedule(epoch)
        
        # Get the current optimizer configuration
        optimizer_config = self.model.optimizer.get_config()

        # Recreate the Adam optimizer with the new momentum (beta_1)
        new_optimizer = Adam(
            learning_rate=optimizer_config['learning_rate'],
            beta_1=new_momentum,  # Update the momentum
            beta_2=optimizer_config['beta_2'],
            epsilon=optimizer_config['epsilon'],
            amsgrad=optimizer_config['amsgrad'],
        )

        # Recompile the model with the new optimizer
        self.model.compile(
            optimizer=new_optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics,
        )
        print(f"Epoch {epoch+1}: Momentum (beta_1) is now {new_momentum}")



# Example of a momentum schedule function
def momentum_schedule(epoch):
    initial_momentum = 0.9
    drop = 0.05
    epochs_drop = 10.0
    return initial_momentum * (1.0 / (1.0 + drop * (epoch // epochs_drop)))

# Define the Adam optimizer with an initial momentum (beta_1)
def create_adam_optimizer(initial_learning_rate=0.001, initial_momentum=0.9):
    return tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        beta_1=initial_momentum,  # Initial momentum (beta_1)
        beta_2=0.999,  # Default value for beta_2
        epsilon=1e-07  # Default value for epsilon
    )

# Example of using the Adam optimizer when compiling the model
def compile_model(model, model_name, use_dynamic_lr, initial_lr=0.001, initial_momentum=0.9):
    # Create an Adam optimizer instance
    adam_optimizer = create_adam_optimizer(initial_learning_rate=initial_lr, initial_momentum=initial_momentum)
    
    if model_name == 'mobilenet_model':
        model.compile(optimizer=adam_optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    else:
        model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Train the model
def train_model(model, train_images, train_masks, use_dynamic_lr=False, epochs=50, batch_size=32, validation_split=0.2):
    # Callbacks list
    callbacks = []

    # Optionally add the dynamic learning rate scheduler callback
    if use_dynamic_lr:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(dynamic_lr)
        callbacks.append(lr_scheduler)

        # Add ReduceLROnPlateau callback for reducing LR based on validation loss
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        callbacks.append(reduce_lr)

        # # Add the dynamic momentum callback
        # dynamic_momentum_callback = DynamicMomentum(model, momentum_schedule)
        # callbacks.append(dynamic_momentum_callback)

    # Train the model
    history = model.fit(
        train_images, train_masks, epochs=150, batch_size=batch_size, 
        validation_split=validation_split, callbacks=callbacks
    )
    return history


# Save the trained model
def save_model(model, model_name, save_path, seed):
    save_full_path = f'{save_path}/{model_name}_224_{seed}.keras'
    model.save(save_full_path)
    print("**********")
    print(f"\n{model_name} model saved at: {save_full_path}")

# Train and save each model
def train_and_save_model(model, model_name, train_images, train_masks, save_path, use_dynamic_lr, seed):
    model = compile_model(model, model_name, use_dynamic_lr)
    print("**********")
    print(f"\nTraining {model_name} model...")

    train_model(model, train_images, train_masks, use_dynamic_lr)

    save_model(model, model_name, save_path, seed)

# Main function to run everything
def main(image_path, mask_path, save_path, use_dynamic_lr, seed):
    print("use_dynamic_lr ::::", use_dynamic_lr)
    
    # Load the data
    train_images, train_masks = load_data(image_path, mask_path)

    # Clear session to avoid issues with model state
    tf.keras.backend.clear_session()

    # List of models to train
    models = {
        # 'attention_unet': attention_unet(input_shape=(224, 224, 3)),
        # 'unet_v3': unet_v3(input_shape=(224, 224, 3)),
        'unet_model': unet_model(),  # Assuming 1 output channel for binary segmentation
        'mobilenet_model': mobilenet_model(output_channels=3),
    }

    # Train and save each model
    for model_name, model in models.items():
        save_path_model = save_path + f"/{model_name}"
        tf.keras.backend.clear_session()  # Clear the session before training each model
        train_and_save_model(model, model_name, train_images, train_masks, save_path_model, use_dynamic_lr, seed)

# Entry point: takes file paths and save path from command-line arguments
if __name__ == "__main__":
    seeds = [40]
    for i in seeds:
        print("Seeds Selected :: ", i)
        gc.collect()
        set_seed(i)  # Set seed for reproducibility
        parser = argparse.ArgumentParser(description='Train segmentation models.')
        # Add arguments for image, mask, save paths, and dynamic learning rate option
        parser.add_argument('--image_path', type=str, required=True, help='Path to the training images (npy file).')
        parser.add_argument('--mask_path', type=str, required=True, help='Path to the training masks (npy file).')
        parser.add_argument('--save_path', type=str, required=True, help='Directory path to save the trained models.')
        parser.add_argument('--use_dynamic_lr', action='store_true', help='Use dynamic learning rate scheduling.')

        args = parser.parse_args()

        # Call main function with user-provided paths and dynamic LR option
        main(args.image_path, args.mask_path, args.save_path, args.use_dynamic_lr, i)
