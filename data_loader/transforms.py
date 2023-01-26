import torch
import torchvision.transforms as transforms


def build_transforms(cfg, is_training):
  
    if is_training:
        transforms_frames = transforms.Compose([
            #transforms.RandomContrast(0.5, 1.5),
            transforms.Resize(cfg['im_size']),
            transforms.ToTensor(),
        ])
        
        transforms_motion = transforms.Compose([
            #transforms.RandomContrast(0.5, 1.5),
            transforms.Resize(cfg['im_size']),
            transforms.ToTensor(),
        ])
        
        transforms_crop = transforms.Compose([
            #transforms.RandomContrast(0.5, 1.5),
            transforms.Resize(cfg['im_size']),
            transforms.ToTensor(),
        ])
    else:
        transforms_frames = transforms.Compose([
            transforms.Resize(cfg['im_size']),
            transforms.ToTensor(),
        ])
        
        transforms_motion = transforms.Compose([
            #transforms.RandomContrast(0.5, 1.5),
            transforms.Resize(cfg['im_size']),
            transforms.ToTensor(),
        ])
        
        transforms_crop = transforms.Compose([
            #transforms.RandomContrast(0.5, 1.5),
            transforms.Resize(cfg['im_size']),
            transforms.ToTensor(),
        ])
        
    return transforms_frames, transforms_motion, transforms_crop