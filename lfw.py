
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np
from tqdm import tqdm
from tools.config import load_config
import torch
from facenet import get_model
from torch.nn import functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 用于将距离转化为相似度
def gaussian(x, sigma=1.0):
    return np.exp(- (x**2) / (2 * sigma**2))

# 计算准确率
def calculate_accuracy(distances, labels, threshold):
    predictions = distances > threshold
    accuracy = accuracy_score(labels, predictions)
    return accuracy

# 找到最佳阈值
def find_best_threshold(distances, labels):
    fpr, tpr, thresholds = roc_curve(labels, distances)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_threshold

def calculate_distances(embeddings):
    distances = []
    for (embed1, embed2) in embeddings:
        dist = F.pairwise_distance(embed1, embed2)  # 欧氏距离
        distances.append(dist.cpu().numpy())
    return np.array(distances)




def eval(trainer, dataLoader, device, writer, epoch):
    trainer.eval()
    embeddings = []
    labels = []
    for batch in tqdm(dataLoader):
        with torch.no_grad():
            label: torch.Tensor
            img_a, img_b, label = batch
            outputs_a = trainer.model_forward(img_a.to(device))
            outputs_b = trainer.model_forward(img_b.to(device))
            embeddings.append((outputs_a, outputs_b))
            labels.append(label.cpu().numpy().astype(np.uint8))
    
    labels = np.array(labels)
    
    # 计算验证结果
    distances: np.ndarray = calculate_distances(embeddings)
    distances = distances.flatten()
    
    # 使用高斯函数进行处理
    distances = gaussian(distances)
    labels = labels.flatten()
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, distances)
    
    # 计算 AUC
    roc_auc = auc(fpr, tpr)
    
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    # 计算准确率
    accuracy = calculate_accuracy(distances, labels, best_threshold)
    
    # 将准确率写入 TensorBoard
    writer.add_scalar("lfw/accuracy", accuracy, epoch)
    
    # 将 AUC 写入 TensorBoard
    writer.add_scalar("lfw/roc_auc", roc_auc, epoch)
    
    return accuracy
    
    # # 绘制 ROC 曲线
    # fig, ax = plt.subplots()
    # ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title('Receiver Operating Characteristic')
    # ax.legend(loc='lower right')
    
    # # 使用 FigureCanvas 将图像绘制到内存中
    # canvas = FigureCanvas(fig)
    # canvas.draw()

    # # 将图像转换为 NumPy 数组
    # img_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 将形状转换为 (H, W, 3)
    
    # # 关闭 plt 图像以释放内存
    # plt.close(fig)

    # # 将 NumPy 数组转换为 TensorBoard 所需的格式 (C, H, W)
    # img_tensor = torch.tensor(img_np).permute(2, 0, 1)

    # # 将 ROC 图像写入 TensorBoard
    # writer.add_image(f'ROC/epoch_{epoch}', img_tensor, epoch)
def main(config):
    
    device = config['eval.device']
    input_shape = config['input_shape']
    batch_size = config['eval.batch_size']
    model = get_model(config['backbone'], "eval", config['eval.model_path'])
    model.eval()
    
    model = model.to(device)

    lfw_path = config['eval.lfw.path']
    pair_path = config['eval.lfw.pair_path']

    lfw_dataset = LFWDataset(lfw_path, pair_path, input_shape)
        
    lfw_dataloader = torch.utils.data.DataLoader(lfw_dataset, batch_size)

    eval(model, lfw_dataloader, device)


if __name__ == "__main__":
    main(load_config("application.yaml"))