import cv2
import torch
import numpy as np
from unet import UNet
from scnn import SCNN
from yolov7.models.yolo import Model
from deep_sort import initialize_deep_sort, run_deep_sort
import sys
import os

# Use raw string literals or double backslashes to avoid escape sequence issues
sys.path.append(r'C:\Users\admin\Desktop\Laneobgandspeedproject\yolov7')
sys.path.append(r'C:\Users\admin\Desktop\Laneobgandspeedproject\yolov7\models')
sys.path.append(r'C:\Users\admin\Desktop\Laneobgandspeedproject\deep_sort\tools')

from deep_sort.tools.generate_detections import create_box_encoder
from yolov7.utils.datasets import letterbox

def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    return frame

def postprocess_and_visualize(frame, lane_mask):
    mask = lane_mask.squeeze().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    frame[mask > 0] = [0, 255, 0]
    return frame

def calculate_speed(position1, position2, time_interval):
    distance = np.sqrt((position2[0] - position1[0])**2 + (position2[1] - position1[1])**2)
    speed = distance / time_interval
    return speed * 3.6  # Convert m/s to km/h

def check_lane_violation(lane_mask, bbox):
    x, y, w, h = bbox
    mask_section = lane_mask[y:y+h, x:x+w]
    violation = np.any(mask_section > 0.5)
    return violation

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    unet = UNet(3, 1).to(device)
    scnn = SCNN(1, 1).to(device)
    yolo = Model('yolov7.yaml').to(device)

    unet.load_state_dict(torch.load(r'unet100.py'))
    scnn.load_state_dict(torch.load(r'scnn100_script.py'))
    yolo.load_state_dict(torch.load(r'v7100.py'))

    encoder, tracker = initialize_deep_sort()
    time_interval = 1 / 30  # Assuming 30 FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame).to(device)
        lane_mask = unet(preprocessed_frame)
        lane_mask = scnn(lane_mask)
        
        detections = yolo(preprocessed_frame)[0]  # assuming yolov7 returns a tuple

        tracker = run_deep_sort(encoder, tracker, frame, detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if track.start_pos is None:
                track.start_pos = (int(bbox[0]), int(bbox[1]))
            else:
                speed = calculate_speed(track.start_pos, (int(bbox[0]), int(bbox[1])), time_interval)
                track.start_pos = (int(bbox[0]), int(bbox[1]))
                cv2.putText(frame, f'Speed: {speed:.2f} km/h', (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if check_lane_violation(lane_mask, (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))):
                cv2.putText(frame, 'Violation!', (int(bbox[0]), int(bbox[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame = postprocess_and_visualize(frame, lane_mask)
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = input("Enter the path to the video file: ")
    main(video_path)
