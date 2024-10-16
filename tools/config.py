
import yaml
from typing import Dict
from warnings import warn

def load_config(file: str):
    with open(file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return Config(config) 
 
class ConfigBinder:
    def __init__(self, config_dict):
        self.bind(config_dict)

    def bind(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用绑定
                setattr(self, key, ConfigBinder(value))
            else:
                # 否则直接设置属性
                setattr(self, key, value)

class Config:
    
    def __init__(self, dict):
        self.data: Dict = dict
    
    def __getitem__(self, key: str):
        res = self.data
        for i in key.split("."):
            res = res.get(i)
            if res is None:
                warn(f"{key} not found !!!")
                return None
        return res