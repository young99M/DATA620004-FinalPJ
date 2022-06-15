import argparse
from xml.etree.ElementInclude import default_loader
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import _LRScheduler
import ml_collections
import torch 
import os
import numpy as np
from model import *

def get_loader(args):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    trainset = datasets.CIFAR100(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
    testset = datasets.CIFAR100(root='./data',
                                train=False,
                                download=True,
                                transform=transform_test) 
    print('train number:', len(trainset))
    print('test number:', len(testset))

    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              num_workers=2,
                              shuffle=True)
    test_loader = DataLoader(testset,
                             batch_size=args.eval_batch_size,
                             num_workers=2,
                             shuffle=False) 
    print('train_loader:', len(train_loader))
    print('test_loader:', len(test_loader))
                        
    return train_loader, test_loader


def get_config():
    """
    配置transformer模型参数
    Returns the ViT-B/16 configuration.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':16})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for i in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

#实例化模型
def getVisionTransformers_model(args):
    config = get_config()  # 获取模型的配置文件
    num_classes = 100  # 因为是cifar100
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.to(args.device)
    return args, model