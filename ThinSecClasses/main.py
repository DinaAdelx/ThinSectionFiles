# main.py
from data_loader import DataLoaderClass
from augmentation import AugmentationClass
from model import ModelClass
from visualization import VisualizationClass
from torch.utils.data import DataLoader

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
    print("Evaluating the best model...")
    model.load_best_model()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = model.evaluate(test_loader)

    print(metrics)

if __name__ == "__main__":
    main()