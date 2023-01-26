from torchvision import datasets, transforms
from base import BaseDataLoader

from datasets import NLCityFlowDataset, build_dataset
from transforms import build_transforms
      
        
class NLCityFlowDataLoader(BaseDataLoader):
    """NL CityFlow data loading demo using BaseDataLoader

    Args:
        BaseDataLoader (_type_): _description_
    """
    def __init__(self, cfg, training, collate_fn=None):
        self.cfg = cfg
        self.training = training
        self.transforms = build_transforms(self.cfg, self.training)
        self.dataset = build_dataset(cfg, self.transforms, self.training)
        self.batch_size = cfg['batch_size']
        self.shuffle = True if self.training else False
        self.validation_split = 0.0
        self.num_workers = cfg['num_workers']
        self.collate_fn = collate_fn
        
        super().__init__(self.dataset, self.batch_size, self.shuffle, self.validation_split, self.num_workers, self.collate_fn)
