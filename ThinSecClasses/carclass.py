from dataset import Dataset
from augmentation import Augmentation
from model import SegmentationModel
from visualization import Visualization
from torch.utils.data import DataLoader

def main():
    # Define paths
    DATA_DIR = './data/CamVid/'
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')
    
    # Create datasets
    model = SegmentationModel()
    preprocessing_fn = model.preprocessor

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=Augmentation.get_training_augmentation(), 
        preprocessing=Augmentation.get_preprocessing(preprocessing_fn),
        classes=['car'],
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=Augmentation.get_validation_augmentation(), 
        preprocessing=Augmentation.get_preprocessing(preprocessing_fn),
        classes=['car'],
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Train model
    model.train(train_loader, valid_loader, epochs=1)
    # Continue with the rest of the main function...
