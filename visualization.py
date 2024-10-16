
import numpy as np
from tools.config import Config, load_config
from torch.utils.data import DataLoader
from tools.train import set_seed
from dataset.ContrastLossDataset import VisualizationDataset, v_dataset_collate
from facenet import FaceNet 
import matplotlib.pylab as plt 
import matplotlib
import torch 

# 设置字体为支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号




def get_result(config: Config):

    seed = config['seed']
    dataset_path = config['eval.dataset_path']
    batch_size = config['eval.batch_size']
    num_workers = config['eval.num_workers']
    backbone = config['backbone']
    set_seed(seed)
    input_shape = config['input_shape']

    device = config['eval.device'] 

    model_path = config['eval.model_path']

    model = FaceNet(backbone, "eval")
    model.load_state_dict(torch.load(model_path)['model'], strict=False)
    model = model.to(device)

    model.eval()
 
    # 获取数据集
    val_dataset     = VisualizationDataset(dataset_path, input_shape)

    val_dataLoader = DataLoader(val_dataset,
                                batch_size = batch_size, 
                                num_workers = num_workers, 
                                pin_memory= True, 
                                drop_last = True, 
                                collate_fn = v_dataset_collate)
    
    data = []
    labels = []
    for imgs, label in val_dataLoader:
        imgs = imgs.to(device)
        
        with torch.no_grad():
            vectors = model(imgs).cpu().numpy()
        
        data.append(vectors)
        labels.append(label.numpy())
    
    data = np.vstack(data)
    labels = np.concatenate(labels)

    return data, labels

def show(data, labels):
    import seaborn as sns
    from sklearn.decomposition import PCA

    # 创建 PCA 对象并将数据降维到 2 维
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # 使用 Seaborn 的调色板
    unique_labels = np.unique(labels)  # 获取唯一的标签
    palette = sns.color_palette("husl", len(unique_labels))  # 使用 Seaborn 调色板

    plt.figure(figsize=(10, 8))

    # 绘制每个类别的散点图
    for i, label in enumerate(unique_labels):
        plt.scatter(data_pca[labels == label, 0], data_pca[labels == label, 1], 
                    color=palette[i], label=f'Class {label}', alpha=0.7)

    plt.title("PCA Projection with Classes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    plt.show()

    plt.waitforbuttonpress()


if __name__ == "__main__":
    from tools.config import load_config

    data, labels = get_result(load_config("application.yaml"))

    show(data, labels)