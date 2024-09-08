import argparse
import numpy as np
import tensorflow as tf
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
def train_model(model, train_images, train_masks, epochs=50, batch_size=32, validation_split=0.2):
    # Add learning rate scheduler and ReduceLROnPlateau callbacks
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(dynamic_lr)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    callbacks = [reduce_lr]

    # Train the model
    history = model.fit(
        train_images, train_masks, epochs=epochs, batch_size=batch_size, 
        validation_split=validation_split, callbacks=callbacks
    )
    return history

# Save the trained model
def save_model(model, model_name):
    save_path = f'../model/{model_name}/{model_name}_224_augmented.keras'
    model.save(save_path)
    print(f"\n{model_name} model saved at: {save_path}")

# Train and save each model
def train_and_save_model(model, model_name, train_images, train_masks):
    model = compile_model(model)
    print(f"\nTraining {model_name} model...")
    model.summary()

    train_model(model, train_images, train_masks)

    save_model(model, model_name)

# Main function to run everything
def main(image_path, mask_path, save_path):
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
        train_and_save_model(model, model_name, train_images, train_masks, save_path)

# Entry point: takes file paths and save path from command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation models.')
    
    # Add arguments for image, mask, and save paths
    parser.add_argument('--image_path', type=str, required=True, help='Path to the training images (npy file).')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the training masks (npy file).')
    parser.add_argument('--save_path', type=str, required=True, help='Directory path to save the trained models.')

    args = parser.parse_args()

    # Call main function with user-provided paths
    main(args.image_path, args.mask_path, args.save_path)
