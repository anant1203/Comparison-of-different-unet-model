#!/bin/bash

# Define variables for model path, test image path, mask image path, target size, and threshold
MODEL_PATH="../../model/unet_original_model/unet_224_original_dynamic_augemented.keras"
TEST_IMG_PATH="../../../numpy_arr_data/augmented_test_images.npy"
MASK_IMG_PATH="../../../numpy_arr_data/augmented_test_masks.npy"
TARGET_SIZE="224 224"  # Provide width and height as two separate values
THRESHOLD=0.5

# Run the Python script with the specified arguments
python testing_script.py \
  --model_path "$MODEL_PATH" \
  --test_img_path "$TEST_IMG_PATH" \
  --mask_img_path "$MASK_IMG_PATH" \
  --target_size $TARGET_SIZE \
  --threshold $THRESHOLD
