import os
import json
import shutil
from tqdm import tqdm

def coco_to_yolo(coco_json_path, output_dir, image_dir):
    # Load the COCO dataset
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create directories for YOLO format
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Map categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        f.write('\n'.join(categories.values()))

    # Create data.yaml file
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(data_yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(output_dir, 'images', 'train')}\n")
        f.write(f"val: {os.path.join(output_dir, 'images', 'val')}\n")
        f.write(f"nc: {len(categories)}\n")
        f.write("names:\n")
        for idx, class_name in categories.items():
            f.write(f"  {idx}: '{class_name}'\n")
    
    print(f"data.yaml file created at {data_yaml_path}")

    # Process each image and its annotations
    for image in tqdm(coco_data['images'], desc="Processing images"):
        img_id = image['id']
        img_name = image['file_name']  # e.g., batch_1/000006.jpg
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        # Extract image subdirectory (e.g., batch_1) from the image path
        img_subdir = os.path.dirname(img_name)  # This will get 'batch_1' for 'batch_1/000006.jpg'

        # Create corresponding subdirectories in 'images' and 'labels'
        image_output_dir = os.path.join(output_dir, "images", img_subdir)
        label_output_dir = os.path.join(output_dir, "labels", img_subdir)

        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        # Copy the image to the output directory
        src_path = os.path.join(image_dir, img_name)
        dst_path = os.path.join(output_dir, "images", img_name)
        
        if not os.path.exists(src_path):
            print(f"Image {img_name} not found at {src_path}. Skipping.")
            continue

        # Ensure the parent directory of the destination file exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

        # Create the corresponding YOLO label file
        label_path = os.path.join(label_output_dir, f"{os.path.splitext(os.path.basename(img_name))[0]}.txt")
        
        with open(label_path, 'w') as label_file:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                width = bbox[2]
                height = bbox[3]
                img_width = image['width']
                img_height = image['height']

                # Normalize coordinates for YOLO format
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height

                # Write the label in YOLO format: class_id x_center y_center width height
                label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    print(f"YOLO dataset saved to {output_dir}")

# Call the function
coco_to_yolo(
    r"C:\Users\sarit\Downloads\tacotrashdataset\dataset.json",  # Path to COCO JSON
    r"C:\Users\sarit\Downloads\tacotrashdataset\dataset_yolo",   # Output directory for YOLO dataset
    r"C:\Users\sarit\Downloads\tacotrashdataset\data"           # Path to images
)
