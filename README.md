# Comparison-of-different-unet-model
This repository provides a shell script that allows you to train and test multiple segmentation models, including UNet and MobileNet variants. You can choose to run the training, testing, or both based on your needs. Additionally, there is an option to use a dynamic learning rate during the training process.

## Prerequisites
Before running the scripts, make sure you have the following installed:

- Python 3.x

- TensorFlow

- NumPy

Ensure that your training and testing data (images and masks) are prepared in .npy format, and the paths to these files are correctly set in the script.


## How to Use the Shell Script
The shell script supports three operations:

1. Train the model(s)
2. Test the model(s)
3. Run both training and testing sequentially

### Shell Script Usage
```
./train_and_test.sh [train|test|both] [options]
```

### Options
- train: Run only the training process.
- test: Run only the testing process.
- both: Run both training and testing sequentially.
- --use_dynamic_lr: (Optional) Use dynamic learning rate during training. If specified, the learning rate will reduce by a factor of 0.5 every 10 epochs.

### Examples

1. Run Training Only:
``` ./train_and_test.sh train ```

2. Run Testing Only:
``` ./train_and_test.sh test ```

3. Run Both Training and Testing:
``` ./train_and_test.sh both ```

4. Run Training with Dynamic Learning Rate:
``` ./train_and_test.sh train --use_dynamic_lr ```

5. Run Both Training and Testing with Dynamic Learning Rate:
``` ./train_and_test.sh both --use_dynamic_lr ```

### Script Breakdown
The shell script automates the following steps:

#### Training:
- Loads training images and masks from .npy files.
- Trains multiple models (e.g., UNet, MobileNet, etc.).
- Optionally applies a dynamic learning rate scheduler that reduces the learning rate every 10 epochs.

#### Testing:
- Loads testing images and masks from .npy files.
- Loads the trained model and evaluates its performance on the test data.
- Compares the model's predictions with the true masks using the specified threshold for binary classification.
- Training Data and Model Paths
- You can customize the paths for your training and testing data as well as the model save directory directly within the script. Make sure the paths match your setup.
