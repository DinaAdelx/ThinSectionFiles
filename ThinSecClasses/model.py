import torch
import segmentation_models_pytorch as smp

class SegmentationModel:
    def __init__(self, encoder='se_resnext50_32x4d', encoder_weights='imagenet', classes=['car'], activation='sigmoid', device='cpu'):
        self.device = device
        self.model = smp.FPN(
            encoder_name=encoder, 
            encoder_weights=encoder_weights, 
            classes=len(classes), 
            activation=activation,
        ).to(device)
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
                torch.save(self.model, './best_model.pth')
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
