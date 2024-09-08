import numpy as np
import tensorflow as tf
import argparse
from model.unet_v3 import unet_v3
from model.unet import unet_model
from model.mobile_net import mobilenet_model
from model.attention_unet import attention_unet

# Load training data
def load_data(image_path, mask_path):
    train_images = np.load(image_path)
    train_masks = np.load(mask_path)
    print(f"Train images shape: {train_images.shape}, Train masks shape: {train_masks.shape}")
    return train_images, train_masks

# Define dynamic learning rate function
def dynamic_lr(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.5  # Reduce learning rate every 10 epochs
    return lr

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

    # Train the model
    history = model.fit(
        train_images, train_masks, epochs=epochs, batch_size=batch_size, 
        validation_split=validation_split, callbacks=callbacks
    )
    return history

# Save the trained model
def save_model(model, model_name, save_path):
    save_full_path = f'{save_path}/{model_name}_224_augmented.keras'
    model.save(save_full_path)
    print(f"\n{model_name} model saved at: {save_full_path}")

# Train and save each model
def train_and_save_model(model, model_name, train_images, train_masks, save_path, use_dynamic_lr):
    model = compile_model(model)
    print(f"\nTraining {model_name} model...")
    model.summary()

    train_model(model, train_images, train_masks, use_dynamic_lr)

    save_model(model, model_name, save_path)

# Main function to run everything
def main(image_path, mask_path, save_path, use_dynamic_lr):
    # Load the data
    train_images, train_masks = load_data(image_path, mask_path)

    # Clear session to avoid issues with model state
    tf.keras.backend.clear_session()

    # List of models to train
    models = {
        'attention_unet': attention_unet(input_shape=(224, 224, 3)),
        'unet_v3': unet_v3(input_shape=(224, 224, 3)),
        'unet_model': unet_model(),  # Assuming 1 output channel for binary segmentation
        'mobilenet_model': mobilenet_model(output_channels=3),
    }

    # Train and save each model
    for model_name, model in models.items():
        tf.keras.backend.clear_session()  # Clear the session before training each model
        train_and_save_model(model, model_name, train_images, train_masks, save_path, use_dynamic_lr)

# Entry point: takes file paths and save path from command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation models.')
    
    # Add arguments for image, mask, save paths, and dynamic learning rate option
    parser.add_argument('--image_path', type=str, required=True, help='Path to the training images (npy file).')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the training masks (npy file).')
    parser.add_argument('--save_path', type=str, required=True, help='Directory path to save the trained models.')
    parser.add_argument('--use_dynamic_lr', action='store_true', help='Use dynamic learning rate scheduling.')

    args = parser.parse_args()

    # Call main function with user-provided paths and dynamic LR option
    main(args.image_path, args.mask_path, args.save_path, args.use_dynamic_lr)
