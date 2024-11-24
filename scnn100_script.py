import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os

class SCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, out_channels)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv4(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)

class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)) / 255.0
        mask = np.expand_dims(mask, axis=0) / 255.0

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def train_scnn():
    image_dir = r'C:\Users\admin\pythonproject\data_segmented'
    mask_dir = r'C:\Users\admin\pythonproject\data2'
    batch_size = 8
    epochs = 50

    dataset = LaneDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scnn = SCNN(3, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(scnn.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scnn = scnn.to(device)

    for epoch in range(epochs):
        scnn.train()
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = scnn(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}')

    torch.save(scnn.state_dict(), 'scnn_model.pth')
    print("Training complete. Model saved as 'scnn_model.pth'.")

def save_scnn_outputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scnn = SCNN(3, 1).to(device)
    scnn.load_state_dict(torch.load('scnn_model.pth'))
    scnn.eval()

    image_dir = r'C:\Users\admin\pythonproject\data_segmented'
    output_dir = r'C:\Users\admin\pythonproject\data_scnn_output'
    os.makedirs(output_dir, exist_ok=True)

    image_list = os.listdir(image_dir)
    for img_name in image_list:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = scnn(image_tensor)
            output = output.squeeze().cpu().numpy()
            output = (output * 255).astype(np.uint8)
        
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, output)

if __name__ == '__main__':
    # You can call train_scnn() if you need to retrain your model
    # train_scnn()

    # Call this function to save the SCNN outputs
    save_scnn_outputs()
