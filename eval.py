from utils import plot_val_far_curve
from trainer import get_model
from tools.config import load_config
from dataset.ContrastLossDataset import SiameseDataset, dataset_collate
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import torch
config = load_config("application.yaml")


def main():
    seed = config['train.seed']
    dataset_path = config['train.dataset_path']
    batch_size = config['train.batch_size']
    num_workers = config['train.num_workers']
    output = config['train.output']
    input_shape = config['input_shape']
    train_num_sample = config['train.train_num_sample']
    val_num_sample = config['train.val_num_sample']
    device = config['predict.device']

    model = get_model(config)


    val_dataset = SiameseDataset(dataset_path, input_shape, "val")

            
        
    val_dataLoader = DataLoader(val_dataset,
                                    batch_size = batch_size, 
                                    num_workers = num_workers, 
                                    pin_memory= True, 
                                    drop_last = True, 
                                    collate_fn = dataset_collate)


    distances = []
    labels = []

    for imgs, label in val_dataLoader:
        
        outputs = []
        for img in imgs:
            with torch.no_grad():
                ouptut = model(img.to(device)) 
                outputs.append(ouptut)
        distance = F.pairwise_distance(*outputs).cpu().numpy()
        distances.append(distance)
        labels.append(label.numpy())

        
    distances = np.concatenate(distances)
    labels = np.concatenate(labels)

    thresholds = np.linspace(min(distances), max(distances), 100)


    plot_val_far_curve(labels, distances, thresholds)


if __name__ == "__main__":
    main()
        


