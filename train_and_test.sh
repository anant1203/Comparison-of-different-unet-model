#!/bin/bash

# Function to display usage instructions
usage() {
  echo "Usage: $0 [train|test|both] [options]"
  echo "  train       Run only the training process"
  echo "  test        Run only the testing process"
  echo "  both        Run both training and testing"
  echo "Options:"
  echo "  --use_dynamic_lr     (Optional) Use dynamic learning rate during training"
  exit 1
}

# Check if at least one argument is passed
if [ $# -lt 1 ]; then
  usage
fi

# Parse the command argument (train, test, or both)
COMMAND=$1
shift  # Shift the command argument out, leaving options

# Check for the dynamic learning rate option
USE_DYNAMIC_LR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --use_dynamic_lr)
      USE_DYNAMIC_LR="--use_dynamic_lr"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Paths for training data, testing data, and model saving
TRAIN_IMG_PATH="../numpy_arr_data/augmented_images.npy"
TRAIN_MASK_PATH="../numpy_arr_data/augmented_masks.npy"
MODEL_SAVE_PATH="../script/model"
TEST_IMG_PATH="../numpy_arr_data/augmented_test_images.npy"
MASK_IMG_PATH="../numpy_arr_data/augmented_test_masks.npy"
TARGET_SIZE="224 224"
THRESHOLD=0.5

# Function to run training
run_training() {
  echo "Starting model training..."
  python src/train_model/train_model.py \
    --image_path "$TRAIN_IMG_PATH" \
    --mask_path "$TRAIN_MASK_PATH" \
    --save_path "$MODEL_SAVE_PATH" \
    $USE_DYNAMIC_LR
  echo "Training completed."
}

# Function to run testing
run_testing() {
  MODEL_PATH="$MODEL_SAVE_PATH/unet_224_augmented.keras"  # Change the model name as needed

  echo "Starting model testing..."
  python testing_script.py \
    --model_path "$MODEL_PATH" \
    --test_img_path "$TEST_IMG_PATH" \
    --mask_img_path "$MASK_IMG_PATH" \
    --target_size $TARGET_SIZE \
    --threshold $THRESHOLD
  echo "Testing completed."
}

# Execute based on user demand
case "$COMMAND" in
  train)
    run_training
    ;;
  test)
    run_testing
    ;;
  both)
    run_training
    run_testing
    ;;
  *)
    echo "Invalid command: $COMMAND"
    usage
    ;;
esac
