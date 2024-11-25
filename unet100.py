import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import torch.optim as optim
import torchvision.models as models

# Double convolution block (used in U-Net architecture)
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

# U-Net architecture with a pretrained ResNet encoder
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=True):
        super(UNet, self).__init__()

        # Using a pretrained ResNet18 as the encoder
        self.encoder = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjusting input channels to 3 (RGB)
        self.encoder.fc = nn.Identity()  # Remove the fully connected layer (not needed for segmentation)

        # Decoder part (custom)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        # Final 1x1 convolution to match the output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder.conv1(x)
        enc2 = self.encoder.layer1(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder.layer2(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder.layer3(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.encoder.layer4(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Decoder with proper concatenation
        dec4 = self.upconv4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=True)  # Resize to match enc4
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenate along the channel dimension
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=True)  # Resize to match enc3
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate along the channel dimension
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=True)  # Resize to match enc2
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenate along the channel dimension
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=True)  # Resize to match enc1
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenate along the channel dimension
        dec1 = self.decoder1(dec1)

        # Ensure final output matches target size
        output_size = x.size()[2:]  # Output size should match input size
        final_output = self.final_conv(dec1)
        output = F.interpolate(final_output, size=output_size, mode='bilinear', align_corners=True)

        return torch.sigmoid(output)


# Dataset class for loading image-mask pairs
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


# Training function
def train_unet():
    image_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\training\kaggle\working\tusimple_preprocessed\training\frames'  # Path to your image data
    mask_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\training\kaggle\working\tusimple_preprocessed\training\lane-masks'    # Path to your mask data
    batch_size = 8
    epochs = 50

    dataset = LaneDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    unet = UNet(3, 1, pretrained=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(unet.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")

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

    # Save the trained model
    torch.save(unet.state_dict(), r'C:\Users\admin\Desktop\Laneobgandspeedproject\UNet_50epochs_16batch_5e-3lr_0.995lrd (1).pt')
    print("Training complete. Model saved as 'unet_model.pth'.")


# Inference function to save segmentation results
def save_segmentation_outputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(3, 1).to(device)
    unet.load_state_dict(torch.load(r'C:\Users\admin\Desktop\Laneobgandspeedproject\UNet_50epochs_16batch_5e-3lr_0.995lrd (1).pt'))  # Path to saved model
    unet.eval()

    image_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\inference_images'  # Input images for inference
    output_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\segmentation_images'  # Directory to save output segmentation masks
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
    # Train the U-Net model
    train_unet()

    # Save the segmentation outputs
    save_segmentation_outputs()
