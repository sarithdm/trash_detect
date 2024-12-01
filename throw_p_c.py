import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained or fine-tuned for trash detection)
yolo = YOLO('yolov8s.pt')

# Load video
video_path = r'C:\Users\sarit\Videos\waste_1.mp4'  # Update path to your video
videoCap = cv2.VideoCapture(video_path)

# Function to determine interaction (e.g., throwing trash)
def is_throwing(person_box, trash_box, vehicle_box=None):
    px1, py1, px2, py2 = person_box
    tx1, ty1, tx2, ty2 = trash_box
    
    # Check if trash is near the person or vehicle
    if px2 >= tx1 and tx2 >= px1 and py2 >= ty1 and ty2 >= py1:
        return True  # Person is near trash
    
    if vehicle_box:
        vx1, vy1, vx2, vy2 = vehicle_box
        if vx2 >= tx1 and tx2 >= vx1 and vy2 >= ty1 and ty2 >= vy1:
            return True  # Trash is near a vehicle
    return False

# Function to get bounding box color
def getColours(cls_num):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return colors[cls_num % len(colors)]

while True:
    ret, frame = videoCap.read()
    if not ret:
        print("End of video or failed to capture frame. Exiting...")
        break

    # Resize frame for better performance
    input_frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    results = yolo(input_frame)

    # Containers for detected objects
    people, trash, vehicles = [], [], []

    # Process detections
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                # Categorize objects
                if class_name == 'person':
                    people.append((x1, y1, x2, y2))
                elif class_name in ['bottle', 'can', 'trash']:  # Define trash-like classes
                    trash.append((x1, y1, x2, y2))
                elif class_name in ['car', 'truck']:  # Vehicle classes
                    vehicles.append((x1, y1, x2, y2))

                # Draw bounding boxes
                color = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check interactions for throwing trash
    for person in people:
        for tr in trash:
            # Check if person is throwing trash
            if is_throwing(person, tr):
                cv2.putText(frame, 'Person Throwing Trash Detected!', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for vehicle in vehicles:
        for tr in trash:
            # Check if trash is thrown from vehicle
            if is_throwing(None, tr, vehicle):
                cv2.putText(frame, 'Trash Thrown from Vehicle Detected!', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Trash Detection', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
videoCap.release()
cv2.destroyAllWindows()
