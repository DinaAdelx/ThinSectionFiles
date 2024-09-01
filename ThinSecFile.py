import os
import cv2
import torch
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset as BaseDataset

# DataLoaderClass: Handles loading and preprocessing the data
class DataLoaderClass:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.x_train_dir = os.path.join(data_dir, 'train')
        self.y_train_dir = os.path.join(data_dir, 'trainannot')
        self.x_valid_dir = os.path.join(data_dir, 'val')
        self.y_valid_dir = os.path.join(data_dir, 'valannot')
        self.x_test_dir = os.path.join(data_dir, 'test')
        self.y_test_dir = os.path.join(data_dir, 'testannot')

    class Dataset(BaseDataset):
        CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
                   'tree', 'signsymbol', 'fence', 'car', 
                   'pedestrian', 'bicyclist', 'unlabelled']
        
        def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
            self.augmentation = augmentation
            self.preprocessing = preprocessing
        
        def __getitem__(self, i):
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
            
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
            return image, mask
        
        def __len__(self):
            return len(self.ids)

    def get_datasets(self, classes, augmentation, preprocessing):
        train_dataset = self.Dataset(self.x_train_dir, self.y_train_dir, classes=classes, 
                                     augmentation=augmentation, preprocessing=preprocessing)
        valid_dataset = self.Dataset(self.x_valid_dir, self.y_valid_dir, classes=classes, 
                                     augmentation=augmentation, preprocessing=preprocessing)
        test_dataset = self.Dataset(self.x_test_dir, self.y_test_dir, classes=classes, 
                                    augmentation=augmentation, preprocessing=preprocessing)
        return train_dataset, valid_dataset, test_dataset

# AugmentationClass: Responsible for managing the augmentations
class AugmentationClass:
    @staticmethod
    def get_training_augmentation():
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            albu.RandomCrop(height=320, width=320, always_apply=True),
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            albu.Perspective(scale=(0.05, 0.1), p=0.5),
            albu.OneOf([albu.CLAHE(p=1), albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), 
                        albu.RandomGamma(p=1)], p=0.9),
            albu.OneOf([albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5), albu.Blur(blur_limit=3, p=1), 
                        albu.MotionBlur(blur_limit=3, p=1)], p=0.9),
            albu.OneOf([albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), 
                        albu.HueSaturationValue(p=1)], p=0.9),
        ]
        return albu.Compose(train_transform)

    @staticmethod
    def get_validation_augmentation():
        return albu.Compose([albu.PadIfNeeded(384, 480)])

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    @staticmethod
    def get_preprocessing(preprocessing_fn):
        return albu.Compose([albu.Lambda(image=preprocessing_fn), albu.Lambda(image=AugmentationClass.to_tensor, 
                                                                              mask=AugmentationClass.to_tensor)])

# ModelClass: Encapsulates the model creation, training, and evaluation
class ModelClass:
    def __init__(self, encoder='se_resnext50_32x4d', encoder_weights='imagenet', classes=['car'], activation='sigmoid', device='cpu'):
        self.device = device
        self.model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), activation=activation)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=0.0001)])
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    def train(self, train_loader, valid_loader, epochs=40):
        max_score = 0
        train_epoch = smp.utils.train.TrainEpoch(self.model, loss=self.loss, metrics=self.metrics, optimizer=self.optimizer, 
                                                 device=self.device, verbose=True)
        valid_epoch = smp.utils.train.ValidEpoch(self.model, loss=self.loss, metrics=self.metrics, device=self.device, verbose=True)
        
        for i in range(epochs):
            print(f'\nEpoch: {i}')
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, './best_model.pth')
                print('Model saved!')
                
            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

    def load_best_model(self):
        self.model = torch.load('./best_model.pth')

    def evaluate(self, test_loader):
        test_epoch = smp.utils.train.ValidEpoch(model=self.model, loss=self.loss, metrics=self.metrics, device=self.device)
        return test_epoch.run(test_loader)

# VisualizationClass: Handles all visualizations
class VisualizationClass:
    @staticmethod
    def visualize(**images):
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([]), plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()

    def visualize_augmentations(self, dataset, n=3):
        for i in range(n):
            image, mask = dataset[i]
            self.visualize(image=image, mask=mask.squeeze(-1))

    def visualize_predictions(self, dataset, model, device='cpu', n=5):
        for i in range(n):
            image_vis, (image, gt_mask) = dataset[i][0].astype('uint8'), dataset[i]
            gt_mask = gt_mask.squeeze()
            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            pr_mask = model.predict(x_tensor).squeeze().cpu().numpy().round()
            self.visualize(image=image_vis, ground_truth_mask=gt_mask, predicted_mask=pr_mask)

# Main function to tie everything together
def main():
    # Initialize classes
    data_loader = DataLoaderClass('./data/CamVid/')
    augmentation = AugmentationClass()
    model = ModelClass(device='cpu')
    visualization = VisualizationClass()

    # Get datasets
    train_dataset, valid_dataset, test_dataset = data_loader.get_datasets(
        classes=['car'], 
        augmentation=augmentation.get_training_augmentation(), 
        preprocessing=augmentation.get_preprocessing(model.preprocessing_fn)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Visualize some augmentations
    print("Visualizing augmentations...")
    visualization.visualize_augmentations(train_dataset)

    # Train the model
    print("Training the model...")
    model.train(train_loader, valid_loader, epochs=40)

    # Load and evaluate the best model
    print
