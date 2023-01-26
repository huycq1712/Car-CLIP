import os
import json
import random
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils.extract_nl import prepare_text, return_a_list_of_NPs


class NLCityFlowDataset(Dataset):
    
    def __init__(self, json_path, frame_path, motion_path, transforms, is_training) -> None:
        super().__init__()
        self.json_path = json_path
        self.frame_path = frame_path
        self.motion_path = motion_path
        self.transforms = transforms
        self.is_training = is_training
        
        with open(json_path, 'r') as f:
            self.tracks = json.load(f)
            
        self.list_of_tracks = list(self.tracks.values())
        self.list_of_uuids = list(self.tracks.keys())
        
        self.all_indexs = list(range(len(self.list_of_uuids)))        
        
    def __len__(self):
        return len(self.list_of_uuids)
    
    
    def __getitem__(self, index):
        index = self.all_indexs[index]
        track = self.list_of_tracks[index]
        nl_idx = int(random.uniform(0, len(track['nl'])))
        nl_view_idx = int(random.uniform(0, len(track['nl_other_views'])))
        frame_idx = int(random.uniform(0, len(track['frames'])))
    
        nl = track['nl'][nl_idx]
        if len(track['nl_other_views']) == 0:
            nl_view = track['nl'][-1]
        else:
            nl_view = track['nl_other_views'][nl_view_idx]
            
        car_noun = 'This is ' + extract_np(nl).lower() + '.'
        
        frame = Image.open(os.path.join(self.frame_path, track['frames'][frame_idx][2:]))
        motion = Image.open(os.path.join(self.motion_path, track['frames'][frame_idx][2:]))
        bbox = track['boxes'][frame_idx]
        bbox = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        
        crop = frame.crop(bbox)

        if self.transforms:
            frame = self.transforms[0](frame)
            motion = self.transforms[1](motion)
            crop = self.transforms[2](crop)
        
        return {
            'index': index,
            'crop_image': crop,
            'global_image': frame,
            'motion_image': motion,
            'nl': nl,
            'nl_view': nl_view,
            'car_noun': car_noun,
        }
        
        
def extract_np(nl):
    original_nl = nl
    nl = prepare_text(nl)
    nl = return_a_list_of_NPs(nl)
    if len(nl) == 0:
        return original_nl
    return nl[0]


def build_dataset(cfg, transforms, is_training):
    dataset = NLCityFlowDataset(json_path = cfg['json_path'],
                                frame_path = cfg['frame_path'],
                                motion_path = cfg['motion_path'],
                                transforms = transforms,
                                is_training = is_training)
    
    return dataset
            
            
if __name__ == '__main__':
    from config import aicity_config
    
    NL_CITY_FLOW_DATASET = NLCityFlowDataset(json_path=aicity_config.train_cfg['json_path'],
                                             frame_path=aicity_config.train_cfg['frame_path'],
                                             motion_path=aicity_config.train_cfg['motion_path'],
                                             transforms=None,
                                             is_training=False)
    
    print(len(NL_CITY_FLOW_DATASET))
    print(NL_CITY_FLOW_DATASET[2])
        
        
        
    