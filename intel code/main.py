!pip install cv2
!pip install numpy
!pip install ultralytics
!pip install opencv-python
!pip install matplotlib
!pip install collections
SyntaxError: invalid syntax
>>> import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import time
import hashlib

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')  # You can use 'yolov8n.pt' for a lighter model

# Define the classes you're interested in
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# Set the confidence threshold
CONF_THRESHOLD = 0.3

# Define filtering thresholds
MAX_OBJECT_SIZE = 5000  # Maximum bounding box area for vehicles
MIN_OBJECT_SIZE = 500   # Minimum bounding box area for vehicles to avoid detecting small objects as vehicles
MIN_ASPECT_RATIO = 0.5  # Minimum aspect ratio (height/width)
MAX_ASPECT_RATIO = 3.0  # Maximum aspect ratio (height/width)
MAX_HEIGHT_RATIO = 0.6  # Maximum height ratio (vehicle shouldn't occupy more than this fraction of the frame height)
MAX_WIDTH_RATIO = 0.8   # Maximum width ratio (vehicle shouldn't occupy more than this fraction of the frame width)

def calculate_speed_distance(frame, model, vehicle_classes, prev_detections, fps):
    results = model(frame)
    vehicles = []
    frame_height, frame_width, _ = frame.shape

    for result in results:
        for detection in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection[:6]
            cls = int(cls)
            if conf > CONF_THRESHOLD and model.names[cls] in vehicle_classes:
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = height / width

                # Apply filtering based on size, aspect ratio, and height/width in the frame
                if (MIN_OBJECT_SIZE < area < MAX_OBJECT_SIZE and
                    MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and
                    (y2 / frame_height) < MAX_HEIGHT_RATIO and
                    (x2 / frame_width) < MAX_WIDTH_RATIO):
                    vehicles.append((int(x1), int(y1), int(x2), int(y2), model.names[cls], conf))

    # Calculate speed and distance (simplified example)
    for vehicle in vehicles:
        x1, y1, x2, y2, cls, conf = vehicle
        height = y2 - y1
        distance = 1000 / height  # Example calculation, adjust based on your setup
        speed = calculate_speed(prev_detections, vehicle, fps)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Speed: {speed:.2f} km/h', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Distance: {distance:.2f} m', (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ttc = calculate_ttc(distance, speed)
        issue_warning(ttc, frame, vehicle, speed, distance)

    return frame, vehicles

def calculate_speed(prev_detections, vehicle, fps):
    # Implement speed calculation based on vehicle movement between frames
    x1, y1, x2, y2, cls, conf = vehicle
    if prev_detections:
        for prev_vehicle in prev_detections:
            prev_x1, prev_y1, prev_x2, prev_y2, prev_cls, prev_conf = prev_vehicle
            if prev_cls == cls:
                prev_center_x = (prev_x1 + prev_x2) / 2
                prev_center_y = (prev_y1 + prev_y2) / 2
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Calculate pixel distance moved
                distance_moved = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                # Convert pixel distance to real-world distance (example calculation)
                distance_moved_real = distance_moved / frame.shape[1] * 50  # Assuming frame width represents 50 meters
                speed = distance_moved_real * fps * 3.6  # Convert to km/h
                return speed
    return 0

def calculate_ttc(distance, speed):
    if speed > 0:
        return distance / (speed / 3.6)  # Speed in m/s
    return float('inf')

def issue_warning(ttc, frame, vehicle, speed, distance):
    if 0.5 <= ttc <= 0.7:
        x1, y1, x2, y2, cls, conf = vehicle
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        vehicle_id = hashlib.md5(f'{x1}{y1}{x2}{y2}{cls}'.encode()).hexdigest()[:8]
        warning_text = (f'WARNING: Potential Collision with {cls}!\n'
                        f'Speed: {speed:.2f} km/h\n'
                        f'Distance: {distance:.2f} m\n'
                        f'Time: {timestamp}\n'
                        f'ID: {vehicle_id}')
        y0, dy = 50, 30
        for i, line in enumerate(warning_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Add additional warning mechanisms like sound alerts here

def process_frame(frame, model, vehicle_classes, prev_detections, fps):
    frame, vehicles = calculate_speed_distance(frame, model, vehicle_classes, prev_detections, fps)
    return frame, vehicles

prev_detections = []

# Open video capture
cap = cv2.VideoCapture("/content/WhatsApp Video 2024-07-15 final 4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('outpu1t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, vehicles = process_frame(frame, model, vehicle_classes, prev_detections, fps)
    prev_detections = vehicles

    out.write(frame)
    cv2_imshow(frame)  # Use cv2_imshow instead of cv2.imshow
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
!pip install cv2
!pip install numpy
!pip install ultralytics
!pip install opencv-python
!pip install matplotlib
!pip install collections!pip install cv2
!pip install numpy
!pip install ultralytics
!pip install opencv-python
!pip install matplotlib
!pip install collections
