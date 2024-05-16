import argparse
import copy

import pytorch_lightning as pl
import torch
from PIL import Image
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from pytorch_lightning.callbacks import ModelCheckpoint

from helpers.configurations import OR_4D_DATA_ROOT_PATH, TAKE_SPLIT


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform

        self.all_image_paths = []
        for take_idx in TAKE_SPLIT['train']:
            color_image_path = OR_4D_DATA_ROOT_PATH / f'export_holistic_take{take_idx}_processed' / 'colorimage'
            self.all_image_paths.extend(list(color_image_path.glob('*.jpg')))

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        # load image
        image = Image.open(image_path)
        image = self.transform(image)
        target = 0
        image_name = image_path.name
        return image, target, image_name, str(image_path)


class DINO(pl.LightningModule):
    def __init__(self, SIZE, LR=1e-5):
        super().__init__()
        assert SIZE in ['s', 'b', 'l', 'g']
        self.SIZE = SIZE
        self.LR = LR

        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{SIZE}14_reg')
        input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.LR)
        return optim

    def get_patch_embeddings(self, x):
        return self.teacher_backbone.forward_features(x)['x_norm_patchtokens']

    def get_global_embeddings(self, x):
        return self.teacher_backbone.forward_features(x)['x_norm_clstoken']


def main():
    '''
    For size s: batchsize 64 and learning rate 1e-5 works good
    For size b: batchsize 64 and learning rate 1e-5 works good
    For size l: batchsize 32 and learning rate 1e-5 works good
    For size g: batchsize 4 and learning rate 1e-5 works good
    Returns
    -------

    '''
    # TODO different lr
    # TODO diffferent temp warmup
    # TODO once Small works somehow, we will scale up
    parser = argparse.ArgumentParser(description='DINO Pretraining')
    parser.add_argument('--size', type=str, default='s', help='Size of the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    SIZE = args.size
    LR = args.lr
    BATCHSIZE = args.batchsize
    name = f's_{SIZE}_bs_{BATCHSIZE}_lr_{LR}_100epochs'
    print(f'Running: {name}')
    # define Transform
    # transform = DINOTransform()
    transform = DINOTransform(local_crop_size=98)
    # define dataset
    dataset = PretrainDataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
        num_workers=12,
    )

    # define model
    model = DINO(SIZE=SIZE, LR=LR)

    # wandb logger
    logger = pl.loggers.WandbLogger(project='ssl_pretrain_dino', save_dir='logs', offline=False, name=name)

    # define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='ssl_pretrain/checkpoints/',  # Directory to save checkpoints.
        filename=f'dino_{name}' + '-{epoch:02d}-{train_loss:.2f}',
        monitor='train_loss',  # Specify the metric to monitor.
        save_top_k=1,  # Save the best model based on the monitored metric.
        mode='min',  # 'min' because we want to minimize the loss.
        every_n_epochs=1,  # Save a checkpoint at every epoch.
        auto_insert_metric_name=False,  # Prevent appending the metric name to the checkpoint filename.
    )

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=100, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=50, num_sanity_val_steps=0, benchmark=False,
                         precision='16-mixed')
    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == '__main__':
    main()
