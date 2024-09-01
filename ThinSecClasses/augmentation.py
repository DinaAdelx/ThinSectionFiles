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
