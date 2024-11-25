import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.models as models

# SCNN with Pretrained ResNet encoder, input is original image + U-Net output
class SCNN(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=True):
        super(SCNN, self).__init__()

        # Using a pretrained ResNet18 as the encoder (for deep feature extraction)
        self.encoder = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust for 3 input channels
        self.encoder.fc = nn.Identity()  # Removing fully connected layer (not needed for segmentation)

        # Decoder part (custom)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(128, 64)

        # Final 1x1 convolution to match output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """A helper function to create a block of convolutions + ReLU + BatchNorm"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, unet_output):
        """ x: original input image, unet_output: output from U-Net """
        # Concatenate the U-Net output with the original image along the channel axis
        x = torch.cat((x, unet_output), dim=1)  # Concatenate along channel axis

        # Encoder
        enc1 = self.encoder.conv1(x)
        enc2 = self.encoder.layer1(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder.layer2(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder.layer3(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.encoder.layer4(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Decoder with concatenation
        dec4 = self.upconv4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final output layer
        final_output = self.final_conv(dec1)
        output = F.interpolate(final_output, size=x.size()[2:], mode='bilinear', align_corners=True)
        return torch.sigmoid(output)


# Dataset class for loading image-mask pairs
class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, unet_output_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.unet_output_dir = unet_output_dir  # New directory for U-Net output
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        unet_output_path = os.path.join(self.unet_output_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unet_output = cv2.imread(unet_output_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None or unet_output is None:
            raise FileNotFoundError(f"Data not found for: {img_path}, {mask_path}, or {unet_output_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)) / 255.0
        mask = np.expand_dims(mask, axis=0) / 255.0
        unet_output = np.expand_dims(unet_output, axis=0) / 255.0  # Normalize the U-Net output

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), torch.tensor(unet_output, dtype=torch.float32)


# Training function
def train_scnn():
    image_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\training\kaggle\working\tusimple_preprocessed\training\frames'  # Path to original images
    mask_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\training\kaggle\working\tusimple_preprocessed\training\lane-masks'  # Path to ground truth masks
    unet_output_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\segmentation_images'  # Path to the directory where U-Net outputs are stored
    batch_size = 8
    epochs = 50

    dataset = LaneDataset(image_dir, mask_dir, unet_output_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scnn = SCNN(4, 1, pretrained=True)  # 3 channels from the image + 1 channel from U-Net output
    criterion = nn.BCELoss()
    optimizer = optim.Adam(scnn.parameters(), lr=0.001)

    # Original code:
# scnn.load_state_dict(torch.load(r'C:\Users\admin\Desktop\Laneobgandspeedproject\fcn_pascalvoc_321x321_fp32_20201111.pt'))

# Modified code to load the model to CPU:
    scnn.load_state_dict(torch.load(r'C:\Users\admin\Desktop\Laneobgandspeedproject\fcn_cityscapes_256x512_20201226.pt', map_location='cpu'))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scnn.to(device)


    scnn = scnn.to(device)

    for epoch in range(epochs):
        scnn.train()
        epoch_loss = 0
        for images, masks, unet_outputs in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            unet_outputs = unet_outputs.to(device)

            outputs = scnn(images, unet_outputs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}')

    torch.save(scnn.state_dict(), r'C:\Users\admin\Desktop\Laneobgandspeedproject\fcn_cityscapes_256x512_20201226.pt')
    print("Training complete. Model saved as 'scnn_model.pth'.")


# Function to save SCNN outputs after training
def save_scnn_outputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scnn = SCNN(4, 1).to(device)
    scnn.load_state_dict(torch.load(r'C:\Users\admin\Desktop\Laneobgandspeedproject\fcn_pascalvoc_321x321_fp32_20201111.pt'))
    scnn.eval()

    image_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\training\kaggle\working\tusimple_preprocessed\training\frames'
    unet_output_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\segmentation_images'  # Where U-Net outputs are stored
    output_dir = r'C:\Users\admin\Desktop\Laneobgandspeedproject\scnn_images'
    os.makedirs(output_dir, exist_ok=True)

    image_list = os.listdir(image_dir)
    for img_name in image_list:
        img_path = os.path.join(image_dir, img_name)
        unet_output_path = os.path.join(unet_output_dir, img_name)
        
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
    # Uncomment to train the SCNN model
    # train_scnn()

    # Call this function to save the SCNN outputs
    save_scnn_outputs()
