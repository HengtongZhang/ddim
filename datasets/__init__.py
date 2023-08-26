import os
import torch
import torchvision.transforms as transforms
from datasets.cifar import CIFAR10
from datasets.celeba import CelebA


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.data_dir, "cifar10"),
            train=True,
            download=False,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.data_dir, "cifar10"),
            train=False,
            download=False,
            transform=test_transform,
        )

    elif config.data.dataset == "CELEBA":
        # cx = 89
        # cy = 121
        # x1 = cy - 64
        # x2 = cy + 64
        # y1 = cx - 64
        # y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.data_dir, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=False,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.data_dir, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [   
                        transforms.ToPILImage(),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=False,
            )

        test_dataset = CelebA(
            root=os.path.join(args.data_dir, "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        )
    else:
        dataset, test_dataset = None, None
        raise NotImplementedError

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
