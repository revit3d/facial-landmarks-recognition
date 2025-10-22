import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import seaborn as sns

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from tqdm.auto import tqdm


def generate_heatmap(size, keypoints, sigma):
    h, w = size
    num_points = keypoints.shape[0]
    heatmaps = np.zeros((num_points, h, w), dtype=np.float32)

    for i, (x, y) in enumerate(keypoints):
        # if x < 0 or y < 0 or x >= w or y >= h:
        #     continue
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return heatmaps


class FaceImageDataset(Dataset):
    def __init__(self, image_dir, gt_dir, img_size, transform: v2.Transform, sigma):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.targets = pd.read_csv(gt_dir, index_col='filename')
        self.img_size = img_size
        self.sigma = sigma

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = np.array(Image.open(img_path).convert('RGB'))

        keypoints = self.targets.loc[img_file].to_numpy().reshape(-1, 2)
        transformed = self.transform(image=image, keypoints=keypoints)
        target = transformed['keypoints']

        target = generate_heatmap(self.img_size, target, sigma=self.sigma)
        return transformed['image'], torch.Tensor(target)


def remap_keypoints(keypoints, **_):
    flip_mapping = [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]
    return keypoints[flip_mapping]


def visualize(inputs, targets, outputs=None, n=16):
    idxs = np.random.randint(0, len(inputs), size=n)

    fig, axes = plt.subplots(int(n**0.5), int(n**0.5), figsize=(20, 20))
    axes = np.ravel(axes)

    for i, idx in enumerate(idxs):
        target = targets[idx]
        xs = [target[j] for j in range(0, len(target), 2)]
        ys = [target[j] for j in range(1, len(target), 2)]
        axes[i].imshow(inputs[idx].permute(1, 2, 0).clip(0, 1))
        axes[i].scatter(xs, ys, c='r')
        for j in range(len(xs)):
            axes[i].text(xs[j] + 1, ys[j] + 1, str(j), c='r')
        if outputs is not None:
            output = outputs[idx]
            assert len(outputs) == len(targets), f'{outputs.shape}, {targets.shape}'
            xs = [output[j] for j in range(0, len(output), 2)]
            ys = [output[j] for j in range(1, len(output), 2)]
            axes[i].scatter(xs, ys, c='g')
            for j in range(len(xs)):
                axes[i].text(xs[j] + 1, ys[j] + 1, str(j), c='g')
        axes[i].axis('off')
    return fig


def heatmaps_to_coords(heatmaps):
    b, n, _, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(b, n, -1)
    max_idx = heatmaps_reshaped.argmax(-1)
    y = (max_idx // w).float()
    x = (max_idx % w).float()
    coords = torch.stack((x, y), dim=-1)
    return coords


def visualize_heatmaps(inputs, targets, outputs):
    idx = np.random.randint(0, len(inputs))

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = np.ravel(axes)

    target = heatmaps_to_coords(targets)[idx].flatten(-2, -1).numpy()
    xs = [target[j] for j in range(0, len(target), 2)]
    ys = [target[j] for j in range(1, len(target), 2)]
    axes[0].imshow(inputs[idx].permute(1, 2, 0).clip(0, 1))
    axes[0].scatter(xs, ys, c='r')
    for j in range(len(xs)):
        axes[0].text(xs[j] + 1, ys[j] + 1, str(j), c='r')
    axes[0].axis('off')

    output = outputs[idx]
    for i in range(14):
        heatmap = output[i].numpy()
        sns.heatmap(heatmap, ax=axes[i + 1], cbar=False)
        axes[i + 1].axis('off')
    return fig


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.se = SEBlock(out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        # Add a 1x1 conv if in_ch dimensions don't match for the residual
        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + res


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=14, ch_mul=32):
        super().__init__()

        self.enc1 = UNetBlock(in_ch, ch_mul)
        self.enc2 = UNetBlock(ch_mul, ch_mul * 2)
        self.enc3 = UNetBlock(ch_mul * 2, ch_mul * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(ch_mul * 4, ch_mul * 8)

        self.up3 = nn.Sequential(
            nn.Upsample(size=(25, 25), mode='bilinear', align_corners=True),
            nn.Conv2d(ch_mul * 8, ch_mul * 4, kernel_size=1)
        )
        self.dec3 = UNetBlock(ch_mul * 8, ch_mul * 4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_mul * 4, ch_mul * 2, kernel_size=1)
        )
        self.dec2 = UNetBlock(ch_mul * 4, ch_mul * 2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_mul * 2, ch_mul, kernel_size=1)
        )
        self.dec1 = UNetBlock(ch_mul * 2, ch_mul)

        self.final = nn.Conv2d(ch_mul, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return self.sigmoid(out)  # (B, num_landmarks, H, W)


def weighted_mse_loss(pred, target, weight_factor=10.0):
    weight = torch.ones_like(target)
    weight[target > 0.1] = weight_factor
    loss = torch.mean(weight * (pred - target) ** 2)
    return loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler = None,
        logger: SummaryWriter | None = None,
        swa_model: AveragedModel | None = None,
        device: str | torch.device = 'cpu',
    ):
        self.model = model
        # self.ema_model = swa_utils.AveragedModel(model, avg_fn=lambda avg, new, n: 0.999 * avg + 0.001 * new)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.swa_model = swa_model
        self.device = device

    def step(self, inputs, targets, model=None) -> torch.Tensor:
        if model is None:
            model = self.model
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        output = model(inputs)
        loss = self.criterion(output, targets)
        return loss, output

    @torch.no_grad
    def validate(self, loader, step, ema=False):
        if ema:
            self.ema_model.module.eval()
        else:
            self.model.eval()

        loss_total = 0
        for inputs, targets in loader:
            _, output = self.step(inputs, targets, self.ema_model.module if ema else None)
            target_coo = heatmaps_to_coords(targets.cpu())
            output_coo = heatmaps_to_coords(output.cpu())
            loss_total += nn.functional.mse_loss(target_coo, output_coo)
        loss_total /= len(loader)
        loss_total = loss_total.cpu().item()

        if self.logger:
            self.logger.add_scalar(f'Loss/Validation{'_ema' if ema else ''}', loss_total, global_step=step)

        if ema:
            self.ema_model.module.train()
        else:
            self.model.train()

        return loss_total

    def train(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int):
        self.model.train()
        # self.ema_model.module.train()

        val_losses = []
        ema_val_losses = []
        for epoch in tqdm(range(n_epochs)):
            last_output = None
            for i, (inputs, targets) in enumerate(train_loader):
                loss, last_output = self.step(inputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                # self.ema_model.update_parameters(self.model)
                if self.logger:
                    global_step = epoch * len(train_loader) + i
                    self.logger.add_scalar(
                        'Loss/Train', loss.item(), global_step,
                    )
                    if self.scheduler:
                        self.logger.add_scalar("Scheduler/lr", self.scheduler.get_last_lr()[0], global_step)
                if self.scheduler:
                    self.scheduler.step()
            if self.logger:
                target_coo = heatmaps_to_coords(targets.cpu())
                output_coo = heatmaps_to_coords(last_output.detach().cpu())
                fig = visualize(inputs, target_coo.flatten(1, -1), output_coo.flatten(1, -1))
                self.logger.add_figure('Examples/Faces', fig, epoch)
                fig = visualize_heatmaps(inputs, targets, last_output.detach().cpu())
                self.logger.add_figure('Examples/Heatmaps', fig, epoch)
            if self.swa_model:
                self.swa_model.update_parameters(self.model)

            val_loss = self.validate(val_loader, step=epoch, ema=False)
            # ema_val_loss = self.validate(val_loader, step=epoch, ema=True)
            val_losses.append(val_loss)
            # ema_val_losses.append(ema_val_loss)

        self.model.eval()
        # self.ema_model.module.eval()

        return val_losses, ema_val_losses


def detect(model_path: str, images_path: str) -> dict:
    """
    Load model from `model_path` and make predictions for all images
    in `images_path` in format [image_file] -> [x1, y1, ..., x14, y14].
    """
    pass


def train_detector(
    train_set: list[str],
    val_set: list[str],
    images_path: str,
    fast_train: bool = False,
) -> torch.nn.Module:
    """
    Train model on images from `train_set` and return trained model.
    If `fast_train` is set to `True`, this function runs on cpu,
    ignores logging, uses one thread and makes significantly less train steps.
    """
    pass
