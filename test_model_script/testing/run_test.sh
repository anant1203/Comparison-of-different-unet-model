#!/bin/bash

# Define arrays for seed values and corresponding model paths
SEED_VALUES=(30 40 50 60 70)

# Define model folders
MODEL_FOLDERS=("unet_model" "mobilenet_model" "attention_unet" "unet_v3")

# Define common variables
TEST_IMG_PATH="../../../numpy_arr_data/test_img.npy"
MASK_IMG_PATH="../../../numpy_arr_data/test_vegetation_mask_gray_img.npy"
TARGET_SIZE="224 224"  # Provide width and height as two separate values
THRESHOLD=0.5

# Define log file
LOG_FILE="./model_testing_log.txt"

# Start logging
echo "Starting model testing at $(date)" > "$LOG_FILE"

# Loop through model folders and test for each seed value
for folder in "${MODEL_FOLDERS[@]}"; do
  for i in "${!SEED_VALUES[@]}"; do
    MODEL_PATH="../../model/$folder/${folder}_224_${SEED_VALUES[$i]}.keras"
    echo "Testing $folder model with seed ${SEED_VALUES[$i]}..." | tee -a "$LOG_FILE"

    # Run the Python script for each model with the specified seed, and log the output
    python testing_script.py \
      --model_path "$MODEL_PATH" \
      --test_img_path "$TEST_IMG_PATH" \
      --mask_img_path "$MASK_IMG_PATH" \
      --target_size $TARGET_SIZE \
      --threshold $THRESHOLD | tee -a "$LOG_FILE"
  done
done

# End logging
echo "Model testing completed at $(date)" >> "$LOG_FILE"
