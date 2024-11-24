import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import torch.optim as optim  # Import the optim module

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

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

def train_unet():
    image_dir = r'C:\Users\admin\pythonproject\data'
    mask_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\data2\idd20k_lite\gtFine\val\62'
    batch_size = 8
    epochs = 50

    dataset = LaneDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    unet = UNet(3, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(unet.parameters(), lr=0.001)  # Using optim from torch.optim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = unet.to(device)

    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = unet(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

    torch.save(unet.state_dict(), 'unet_model.pth')
    print("Training complete. Model saved as 'unet_model.pth'.")

def save_segmentation_outputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(3, 1).to(device)
    unet.load_state_dict(torch.load('unet_model.pth'))
    unet.eval()

    image_dir = r'C:\Users\admin\pythonproject\data'
    output_dir = r'C:\Users\admin\pythonproject\data_segmented'
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
            output = unet(image_tensor)
            output = output.squeeze().cpu().numpy()
            output = (output * 255).astype(np.uint8)
        
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, output)

if __name__ == '__main__':
    # Call this function to train and save the U-Net model
    train_unet()

    # Call this function to save the segmentation outputs
    save_segmentation_outputs()
