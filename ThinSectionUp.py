#new file for complete clases

#augmentation.py class 

import albumentations as albu

class Augmentation:
    @staticmethod
    def get_training_augmentation():
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0, value=(0, 0, 0)),
            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0, value=(0, 0, 0)),
            albu.RandomCrop(height=320, width=320, always_apply=True),
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            albu.Perspective(scale=(0.05, 0.1), p=0.5),
            albu.OneOf(
                [albu.CLAHE(p=1),
                 albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                 albu.RandomGamma(p=1)],
                p=0.9,
            ),
            albu.OneOf(
                [albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                 albu.Blur(blur_limit=3, p=1),
                 albu.MotionBlur(blur_limit=3, p=1)],
                p=0.9,
            ),
            albu.OneOf(
                [albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                 albu.HueSaturationValue(p=1)],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    @staticmethod
    def get_validation_augmentation():
        return albu.Compose([albu.PadIfNeeded(384, 480)])

    @staticmethod
    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callable): data normalization function 
                (can be specific for each pretrained neural network)
        """
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=Augmentation.to_tensor, mask=Augmentation.to_tensor),
        ]
        return albu.Compose(_transform)

    @staticmethod
    def to_tensor(x, **kwargs):
        """Convert image or mask to a tensor."""
        return x.transpose(2, 0, 1).astype('float32')
import albumentations as albu

class Augmentation:
    @staticmethod
    def get_training_augmentation():
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0, value=(0, 0, 0)),
            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0, value=(0, 0, 0)),
            albu.RandomCrop(height=320, width=320, always_apply=True),
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            albu.Perspective(scale=(0.05, 0.1), p=0.5),
            albu.OneOf(
                [albu.CLAHE(p=1),
                 albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                 albu.RandomGamma(p=1)],
                p=0.9,
            ),
            albu.OneOf(
                [albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                 albu.Blur(blur_limit=3, p=1),
                 albu.MotionBlur(blur_limit=3, p=1)],
                p=0.9,
            ),
            albu.OneOf(
                [albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                 albu.HueSaturationValue(p=1)],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    @staticmethod
    def get_validation_augmentation():
        return albu.Compose([albu.PadIfNeeded(384, 480)])

    @staticmethod
    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callable): data normalization function 
                (can be specific for each pretrained neural network)
        """
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=Augmentation.to_tensor, mask=Augmentation.to_tensor),
        ]
        return albu.Compose(_transform)

    @staticmethod
    def to_tensor(x, **kwargs):
        """Convert image or mask to a tensor."""
        return x.transpose(2, 0, 1).astype('float32')
    
    
#dataset.py class 


import os
import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    CLASSES = ['Biotite_Gr', 'Biotite2_Gr', 'Biotite3_Gr', 'Chlorite_afterBiotite', 'Chlorite2_after_biotite_', 
               'Muscovite2_clip_redantant_stack', 'Olivine', 'Talc_Serpentinite', 'Talc2_Serpentinite', 
               'Titanite', 'Tourmaline_Gr']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Convert class names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
      # Read image with rasterio
      with rasterio.open(self.images_fps[i]) as src:
          image = src.read()  # read all bands

      # Convert from (C, H, W) to (H, W, C) and then to (C, H, W)
      image = np.transpose(image, (1, 2, 0))
      image = np.transpose(image, (2, 0, 1))  # Now image is (C, H, W)

      # Read mask with rasterio
      with rasterio.open(self.masks_fps[i]) as src:
          mask = src.read(1)  # read the first band for the mask

      # Extract specific classes from mask
      masks = [(mask == v) for v in self.class_values]
      mask = np.stack(masks, axis=-1).astype('float')

      # Apply augmentations only for 1 or 3 channel images
      if self.augmentation:
          if image.shape[0] in [1, 3]:  # Apply Albumentations only for 1 or 3 channel images
              sample = self.augmentation(image=image, mask=mask)
              image, mask = sample['image'], sample['mask']
          else:
              # NumPy-based operations for 6-channel images
              print(f"Applying NumPy-based operations for {image.shape[0]}-channel image")
              
              # Normalize pixel values to [0, 1]
              image = image / 255.0
              
              # Add Gaussian noise
              noise = np.random.normal(0, 0.01, image.shape)
              image = image + noise
      
              # Clip values to stay within valid range [0, 1]
              image = np.clip(image, 0, 1)

      # Ensure mask is (num_classes, H, W) for PyTorch
      mask = np.transpose(mask, (2, 0, 1))  # Ensure mask is (num_classes, H, W)

      # Apply preprocessing
      if self.preprocessing:
          if image.shape[0] == 6:
              # Custom preprocessing for 6-channel images
              image = image / 255.0  # Normalize 6-channel image
          else:
              # Use the standard preprocessing for 3-channel images
              sample = self.preprocessing(image=image, mask=mask)
              image, mask = sample['image'], sample['mask']

      # Convert to PyTorch tensors
      image = torch.tensor(image, dtype=torch.float32)
      mask = torch.tensor(mask, dtype=torch.float32)

      return image, mask



    def __len__(self):
        return len(self.ids)

#class model.py 


import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class CustomSEEncoder(nn.Module):
    def __init__(self, in_channels):
        super(CustomSEEncoder, self).__init__()
        # Load the pre-trained SE-ResNeXt50 model with modified input channels
        self.encoder = smp.encoders.get_encoder('se_resnext50_32x4d', in_channels=in_channels)

    def forward(self, x):
        return self.encoder(x)

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # Initialize the custom encoder with 6 channels
        self.encoder = CustomSEEncoder(in_channels=6)
        
        # Initialize Unet model with the custom encoder
        self.unet = smp.Unet(
            encoder_name='se_resnext50_32x4d', 
            encoder_weights=None,  # No pre-trained weights since we modify the encoder
            in_channels=6, 
            classes=num_classes
        )

        # Assign the custom encoder to the model
        self.unet.encoder = self.encoder.encoder

    def forward(self, x):
        # Ensure input is (batch_size, channels, height, width)
        if x.shape[1] == 6:  # If channels are in the right position
            return self.unet(x)
        else:
            x = x.permute(0, 3, 1, 2)  # Adjust input from (B, H, W, C) to (B, C, H, W)
            return self.unet(x)




def get_model():
    # Load the original encoder
    original_encoder = smp.encoders.get_encoder('se_resnext50_32x4d', encoder_weights='imagenet')
    custom_encoder = CustomEncoder(original_encoder, in_channels=6)

    # Use the custom encoder in your segmentation model
    model = smp.Unet(
        encoder_name='se_resnext50_32x4d', 
        encoder_weights=None,  # No pre-trained weights since we modify the encoder
        in_channels=6, 
        classes=11  # Number of classes in your segmentation task
    )
    model.encoder = custom_encoder  # Set the custom encoder to the model
    
    return model



class SegmentationModel:
    def __init__(self, encoder='se_resnext50_32x4d', encoder_weights='imagenet', classes=None, activation='sigmoid', device='cpu'):
        self.device = device
        if classes is None:
            classes = ['Biotite_Gr', 'Biotite2_Gr', 'Biotite3_Gr', 'Chlorite_afterBiotite', 
                       'Chlorite2_after_biotite_', 'Muscovite2_clip_redantant_stack', 
                       'Olivine', 'Talc_Serpentinite', 'Talc2_Serpentinite', 
                       'Titanite', 'Tourmaline_Gr']
        
        # Initialize custom encoder
        custom_encoder = CustomSEEncoder(in_channels=6)
        
        # Initialize the Unet model with the custom encoder
        self.model = smp.Unet(
            encoder_name=encoder, 
            encoder_weights=None,  # No pre-trained weights since we modify the encoder
            in_channels=6, 
            classes=len(classes), 
            activation=activation
        )
        
        self.model.encoder = custom_encoder.encoder  # Set the custom encoder
        
        self.model.to(device)
        self.preprocessor = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [smp.utils.metrics.IoU(threshold=0.5)]
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=0.0001)])
        
    def train(self, train_loader, valid_loader, epochs=1):
        max_score = 0
        for i in range(epochs):
            print('\nEpoch: {}'.format(i))
            train_epoch = smp.utils.train.TrainEpoch(
                self.model, 
                loss=self.loss, 
                metrics=self.metrics, 
                optimizer=self.optimizer,
                device=self.device,
                verbose=True,
            )
            valid_epoch = smp.utils.train.ValidEpoch(
                self.model, 
                loss=self.loss, 
                metrics=self.metrics, 
                device=self.device,
                verbose=True,
            )   

            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model.state_dict(), './best_model.pth')  # Save model state dict
                print('Model saved!')

            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
    
    def evaluate(self, test_loader):
        test_epoch = smp.utils.train.ValidEpoch(
            model=self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
        )
        logs = test_epoch.run(test_loader)
        return logs

#class visualization.py


import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def visualize(image, mask):
      """Visualize an image and its mask."""
      plt.figure(figsize=(12, 6))
      plt.subplot(1, 2, 1)
      plt.imshow(image)
      plt.title("Image")
      plt.axis('off')

      plt.subplot(1, 2, 2)
      # Assume mask has been stacked along the last dimension
      combined_mask = np.argmax(mask, axis=-1) if mask.ndim == 3 else mask
      plt.imshow(combined_mask)
      plt.title("Mask")
      plt.axis('off')

      plt.show()  


#main.py

import os
from dataset import Dataset
from augmentation import Augmentation
from model import SegmentationModel
from visualization import Visualization
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
import ssl

# Disable SSL certificate verification (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Define paths and other parameters
    DATA_DIR = 'Dataset01-20240815T132523Z-001'
    x_train_dir = os.path.join(DATA_DIR, 'TrainImg')
    y_train_dir = os.path.join(DATA_DIR, 'TrainLabel')
    x_valid_dir = os.path.join(DATA_DIR, 'ValidImg')
    y_valid_dir = os.path.join(DATA_DIR, 'ValidLabel')
    
    CLASSES = ['Biotite_Gr', 'Biotite2_Gr', 'Biotite3_Gr', 'Chlorite_afterBiotite', 'Chlorite2_after_biotite_', 
               'Muscovite2_clip_redantant_stack', 'Olivine', 'Talc_Serpentinite', 'Talc2_Serpentinite', 
               'Titanite', 'Tourmaline_Gr']
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Create datasets
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=Augmentation.get_training_augmentation(), 
        preprocessing=Augmentation.get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )
    
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=Augmentation.get_validation_augmentation(), 
        preprocessing=Augmentation.get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize the model with the correct number of classes
    model = SegmentationModel(
      encoder=ENCODER,
      encoder_weights=ENCODER_WEIGHTS,
      classes=CLASSES,  # Pass the list of classes directly, not len(CLASSES)
      activation=ACTIVATION,
      device=DEVICE
  )



    # Start training
    model.train(train_loader, valid_loader, epochs=1)


    # Continue with additional steps such as testing, saving the model, and evaluation
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # logs = model.evaluate(test_loader)
    # print(logs)

if __name__ == "__main__":
    main()






