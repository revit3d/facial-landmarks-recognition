import torch


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
