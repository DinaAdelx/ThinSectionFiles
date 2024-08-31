# data_loader.py
import os
import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset


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

