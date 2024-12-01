import os
import shutil
import random

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    # Create train and val directories for images and labels in the output directory
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # Get a list of all image files
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, f))

    # Check if any images were found
    if not image_files:
        print("No image files found in the specified directory.")
        return

    # Shuffle the image files to randomize the split
    random.shuffle(image_files)

    # Calculate the number of training images
    num_train = int(len(image_files) * train_ratio)

    # Split the image files into training and validation sets
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    # Move the images and corresponding labels to the respective directories
    for img_file in train_files:
        img_name = os.path.basename(img_file)
        img_subdir = os.path.basename(os.path.dirname(img_file))  # Get subdirectory (e.g., batch_6)

        # Create subdirectories in train for images and labels
        train_img_dir = os.path.join(output_dir, "images", "train", img_subdir)
        train_label_dir = os.path.join(output_dir, "labels", "train", img_subdir)
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)

        # Copy image to train directory
        shutil.copy(img_file, os.path.join(train_img_dir, img_name))

        # Copy corresponding label file to train directory
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_file_path = os.path.join(label_dir, img_subdir, label_file)
        if os.path.exists(label_file_path):
            shutil.copy(label_file_path, os.path.join(train_label_dir, label_file))
        else:
            print(f"Label file missing for image {img_name}. Skipping.")

    for img_file in val_files:
        img_name = os.path.basename(img_file)
        img_subdir = os.path.basename(os.path.dirname(img_file))  # Get subdirectory (e.g., batch_6)

        # Create subdirectories in val for images and labels
        val_img_dir = os.path.join(output_dir, "images", "val", img_subdir)
        val_label_dir = os.path.join(output_dir, "labels", "val", img_subdir)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # Copy image to val directory
        shutil.copy(img_file, os.path.join(val_img_dir, img_name))

        # Copy corresponding label file to val directory
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_file_path = os.path.join(label_dir, img_subdir, label_file)
        if os.path.exists(label_file_path):
            shutil.copy(label_file_path, os.path.join(val_label_dir, label_file))
        else:
            print(f"Label file missing for image {img_name}. Skipping.")

    print(f"Dataset split: {len(train_files)} images for training, {len(val_files)} images for validation.")

# Define paths
coco_json_path = r"C:\Users\sarit\Downloads\tacotrashdataset\dataset.json"  # Path to COCO JSON
image_dir = r"C:\Users\sarit\Downloads\tacotrashdataset\data"  # Path to images
output_dir = r"C:\Users\sarit\Downloads\tacotrashdataset\dataset_yolo"  # Output directory for YOLO dataset

# Define the directory for images and labels after conversion
image_dir = os.path.join(output_dir, "images")
print(image_dir)
label_dir = os.path.join(output_dir, "labels")
print(label_dir)

# Split dataset into training and validation sets
dataset_output_dir = r"C:\Users\sarit\Downloads\tacotrashdataset\dataset_train_test_split"  # Output directory for split dataset
split_dataset(image_dir, label_dir, dataset_output_dir)
