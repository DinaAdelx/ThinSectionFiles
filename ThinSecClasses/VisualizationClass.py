# visualization.py
import matplotlib.pyplot as plt

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
