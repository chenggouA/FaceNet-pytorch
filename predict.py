from PIL import Image
from tools.config import Config
from SiameseNetwork import ModelWrapper
import torch

def get_modelWrapper(config: Config):
    
    mode = config['predict.mode']
    device = config['predict.device']
    num_classes = config['num_classes']
    input_size = config['input_shape']
    
    model_path = config['predict.model_path']
    
    model = ModelWrapper(model_path)

    return model 
    
def main(config: Config):
        
    model = get_modelWrapper(config)
    
    image_path  = config['predict.image_path']
    image: Image.Image = model(r'output\mosaic\mosaic_2.jpg')

    image.show()


if __name__ == "__main__":
    from tools.config import load_config

    main(load_config("application.yaml"))