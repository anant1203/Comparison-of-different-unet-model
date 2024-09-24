import numpy as np
import tensorflow as tf
import cv2
import argparse

def load_model(model_path):
    """
    Load a pre-trained model from the specified path.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def load_data(test_img_path, mask_img_path):
    """
    Load test images and mask images from the specified paths.
    """
    test_image = np.load(test_img_path)
    mask_image = np.load(mask_img_path)
    
    # Add an extra dimension to mask images (for single-channel grayscale masks)
    mask_image = mask_image[..., np.newaxis]
    
    return test_image, mask_image

def resize_images(images, target_size=(224, 224)):
    """
    Resize images to the target size.
    """
    resized_images = [tf.image.resize(image, target_size).numpy() for image in images]
    return np.array(resized_images)

def rgb2gray(rgb):
    """
    Convert RGB images to grayscale.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def convert_to_grayscale(images):
    """
    Convert a list of images from RGB to grayscale.
    """
    gray_images = [rgb2gray(image) for image in images]
    gray_images = np.array(gray_images)[..., np.newaxis]
    return gray_images

def predict_masks(model, test_images):
    """
    Use the model to predict masks for the test images.
    """
    return model.predict(test_images, verbose=1)

def post_process_masks(predicted_masks, threshold=0.5):
    """
    Post-process the predicted masks by applying a binary threshold.
    """
    processed_masks = []
    for mask in predicted_masks:
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        processed_masks.append(mask)
    return np.array(processed_masks)

def calculating_iou(true_masks, pred_masks):
    """
    Calculate the Intersection over Union (IoU) between true and predicted masks.
    """
    iou_scores = []
    for i in range(len(true_masks)):
        intersection = np.logical_and(true_masks[i], pred_masks[i])
        union = np.logical_or(true_masks[i], pred_masks[i])
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)
    
    iou_scores = np.array(iou_scores)
    iou_scores = np.nan_to_num(iou_scores, copy=True, nan=1.0)
    mean_iou = iou_scores.mean()
    return mean_iou

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Model Inference and IoU Calculation")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--test_img_path', type=str, required=True, help='Path to the test images in .npy format')
    parser.add_argument('--mask_img_path', type=str, required=True, help='Path to the mask images in .npy format')
    parser.add_argument('--target_size', type=int, nargs=2, default=(224, 224), help='Target size for image resizing (width height)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for mask binarization')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load test and mask images
    test_images, true_masks = load_data(args.test_img_path, args.mask_img_path)
    
    # Resize images and masks
    test_images_resized = resize_images(test_images, target_size=args.target_size)
    true_masks_resized = resize_images(true_masks, target_size=args.target_size)
    
    # Display the shapes of the resized test images and masks
    print("Test image shape:", test_images_resized.shape)
    print("Mask image shape:", true_masks_resized.shape)
    
    # Convert test images to grayscale
    # gray_test_images = convert_to_grayscale(test_images_resized)
    # print("Gray image shape:", gray_test_images.shape)
    
    # Predict masks
    predicted_masks = predict_masks(model, test_images_resized)
    print("Predicted image shape:", predicted_masks.shape)
    
    # Post-process predicted masks
    processed_pred_masks = post_process_masks(predicted_masks, threshold=args.threshold)
    
    # Calculate IoU
    mean_iou = calculating_iou(true_masks_resized, processed_pred_masks)
    print("Mean IoU score:", mean_iou)

# Run the script
if __name__ == "__main__":
    main()
