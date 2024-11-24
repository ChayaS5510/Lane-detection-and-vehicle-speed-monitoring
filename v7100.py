import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from yolov7.models.yolo import *
from yolov7.utils.datasets import create_dataloader
from yolov7.utils.general import check_dataset

def custom_loss(preds, targets, model):
    return torch.nn.functional.mse_loss(preds, targets)

# Define paths
base_dir = os.path.dirname(os.path.abspath(r'C:\Users\admin\Desktop\Laneobgandspeedproject\data3\JUVC\data\data.yaml'))

data_path = os.path.join(base_dir, '..', 'data3', 'JUVC', 'data', 'data.yaml')
cfg = os.path.join(base_dir, '..', 'yolov7', 'cfg', 'deploy', 'yolov7.yaml')
weights = os.path.join(base_dir, '..', 'yolov7', 'yolov7.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10

# Load model
model = Model(cfg).to(device)
if weights:
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

# Create dataloader
dataloader, dataset = create_dataloader(data_path, img_size=640, batch_size=16, stride=32, single_cls=False)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
all_labels = []
all_preds = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        preds = model(imgs)
        loss = custom_loss(preds, targets, model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        _, predicted = torch.max(preds, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        epoch_loss += loss.item()
        
        all_labels.extend(targets.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
plt.show()

torch.save(model.state_dict(), os.path.join(base_dir, 'trained_yolov7.pth'))

def detect_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)
    output_dir = r'C:\Users\admin\pythonproject\video_output'
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(frame_tensor)

        output_frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(output_frame_path, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = input("Enter the path to the video file: ")
    detect_vehicles(video_path)
