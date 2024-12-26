import os
import json

def convert_labelme_to_yolo_pose(labelme_json_path, output_folder):
    # Load LabelMe JSON
    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    # Extract image properties
    img_width = labelme_data['imageWidth']
    img_height = labelme_data['imageHeight']
    image_name = os.path.splitext(labelme_data['imagePath'])[0]
    shapes = labelme_data['shapes']

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process shapes for annotations
    for shape in shapes:
        label = shape['label']
        points = shape['points']  # List of (x, y) points

        # Convert points to keypoints format (x, y, visibility)
        keypoints = []
        for point in points:
            x_norm = point[0] / img_width
            y_norm = point[1] / img_height
            visibility = 2  # Assuming fully visible keypoints
            keypoints.extend([x_norm, y_norm, visibility])

        # Create YOLO-compatible bounding box around the keypoints
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        bbox_x_center = ((x_min + x_max) / 2) / img_width
        bbox_y_center = ((y_min + y_max) / 2) / img_height
        bbox_width = (x_max - x_min) / img_width
        bbox_height = (y_max - y_min) / img_height

        # Combine into YOLO format
        yolo_line = f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} " + " ".join(
            [f"{kp:.6f}" for kp in keypoints]
        )

        # Save to .txt file
        output_file = os.path.join(output_folder, f"{image_name}.txt")
        with open(output_file, 'a') as f:
            f.write(yolo_line + "\n")

    return f"Conversion complete. Annotations saved in {output_folder}"


# Define paths
labelme_json_path =  r'C:\Users\sarit\Downloads\data\test.json'  # Path to the LabelMe JSON file
output_folder = r'C:\Users\sarit\Downloads\data\output'  # Folder to save YOLO annotations

# Run the conversion
result_message = convert_labelme_to_yolo_pose(labelme_json_path, output_folder)
print(result_message)

