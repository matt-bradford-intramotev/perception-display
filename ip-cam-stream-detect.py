import cv2
from ultralytics import YOLO
import torch
import sys

# RTSP URL of the stream
if len(sys.argv) < 2:
    print("Defaulting to 192.168.1.22 (TrinFlo A)")
    HOST_IP = '192.168.1.22'
else:
    HOST_IP = sys.argv[1]

rtsp_url = f"rtsp://root:tugvolt@{HOST_IP}/axis-media/media.amp"
print(f"RTSP URL: {rtsp_url}")

# Acceptable labels
# LABELS_TO_USE = ['person','car','bus','train','truck','traffic light','stop sign']
LABELS_TO_USE = ['person','car','bus','truck']

# Load YOLOv8 model (replace 'yolov8n.pt' with your desired model weights)
model = YOLO('yolov8m.pt')
# model = YOLO('nuimage100best.pt')

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Set up screen size, etc
screen_width = 3840
screen_height = 1080
quad_width = screen_width // 2
quad_height = screen_height // 2
quad_x = screen_width // 2
quad_y = 0

# Open window
cv2.namedWindow('Camera Stream', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Camera Stream', quad_width, quad_height)
cv2.moveWindow('Camera Stream', quad_x, quad_y)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    try:
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Could not read frame")
            break

        # Perform object detection
        results = model(frame)
        
        # Loop through the results to draw bounding boxes and labels on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label in LABELS_TO_USE:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label and confidence
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Stream', frame)

        # Press 'q' to exit the video display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
