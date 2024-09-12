import os
import cv2
import glob
import json
import imageio
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor

def prepare_id(file_path):
    """Extracts the ID from the filename by removing the suffix '_gtFine_color.png'."""
    filename = os.path.basename(file_path)
    extracted_part = filename.replace('_gtFine_color.png', '')
    return extracted_part

def city_folder(x):
    """Extracts the city name from the file ID."""
    return x.split("_")[0]

def mask_json_file_path(image_path, mask_dir):
    """Generates the mask JSON file path based on the image ID."""
    return os.path.join(mask_dir, image_path + "_gtFine_polygons.json")

def train_image_path(image_path, img_dir):
    """Generates the train image path based on the image ID."""
    return os.path.join(img_dir, image_path + "_leftImg8bit.png")

def images_to_numpy_array(images):
    """Converts a list of images to a NumPy array."""
    return np.array(images)

def get_vegetation_mask(image, data):
    """Creates a mask for vegetation regions."""
    for i in data["objects"]:
        points = np.array(i["polygon"], np.int32)
        points = points.reshape((-1, 1, 2))
        if i['label'] == 'vegetation':
            image = cv2.fillPoly(image, [points], (255, 255, 255))
        else:
            image = cv2.fillPoly(image, [points], (0, 0, 0))
    image[image != 255] = 0
    return image

def process_masks_and_images(mask_dir, img_dir, output_dir):
    # Get all mask files
    mask_files = [f for f in glob.glob(mask_dir + "**/*color.png", recursive=True)]
    print("Number of mask files:", len(mask_files))

    # Prepare DataFrame
    df = pd.DataFrame()
    df["mask_file"] = mask_files
    df["id"] = df.mask_file.map(lambda x: prepare_id(x))
    df["city_name"] = df.id.map(lambda x: city_folder(x))
    df["file_path_suffix"] = df["city_name"] + "/" + df["id"]
    df["mask_json_file_path"] = df.file_path_suffix.map(lambda x: mask_json_file_path(x, mask_dir))
    df["test_image_path"] = df.file_path_suffix.map(lambda x: train_image_path(x, img_dir))
    
    # Load mask images
    train_mask_img = []
    for mask_image_path in mask_files:
        mask_arr = imageio.imread(mask_image_path)
        img = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
        train_mask_img.append(img)
    
    train_mask_img = images_to_numpy_array(train_mask_img)
    
    # Load mask JSON files
    train_mask_json = []
    for mask_json in df.mask_json_file_path.values:
        with open(mask_json, 'r') as file:
            train_mask_json.append(json.load(file))
    
    # Create vegetation masks
    vegetation_mask_img = []
    for img, json_data in zip(train_mask_img, train_mask_json):
        img = get_vegetation_mask(img, json_data)
        vegetation_mask_img.append(img)

    vegetation_mask_img = [tf.image.resize(mask, (224, 224)).numpy() for mask in vegetation_mask_img]
    vegetation_mask_img = images_to_numpy_array(vegetation_mask_img)
    np.save(os.path.join(output_dir, 'train_mask.npy'), vegetation_mask_img)
    
    # Load train images
    train_img = []
    for train_img_path in df.test_image_path.values:
        train_arr = imageio.imread(train_img_path)
        img = cv2.cvtColor(train_arr, cv2.COLOR_BGR2RGB)
        train_img.append(img)

    # Prepare your dataset as before
    train_img = [tf.image.resize(images, (224, 224)).numpy() for images in train_img]
    train_img = images_to_numpy_array(train_img)
    np.save(os.path.join(output_dir, 'train_img.npy'), train_img)
    
    print("All data has been processed and saved.")

def main(mask_dir, img_dir, output_dir):
    """Main function to process mask and image directories."""
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Call the processing function
    process_masks_and_images(mask_dir, img_dir, output_dir)

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Process mask and image data.")
    
    # Add arguments for mask directory, image directory, and output directory
    parser.add_argument('--mask_dir', type=str, required=True, 
                        help="Path to the mask directory.")
    parser.add_argument('--img_dir', type=str, required=True, 
                        help="Path to the train image directory.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the output directory where results will be saved.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with the provided arguments
    main(args.mask_dir, args.img_dir, args.output_dir)
