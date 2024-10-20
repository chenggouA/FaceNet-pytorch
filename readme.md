
# facenet-pytorch

## 主干网络支持

- Inception ResNet v1
- MobileNet v2

## 损失函数支持

- Contrast Loss
- TripletLoss Loss


| 主干网络             | 使用分类辅助 | 训练轮数 | 损失函数   | 下载连接                             | LFW 准确率 |
| -------------------- | ------------ | ------- | ---------- | ------------------------------------ | ---------- |
| MobileNetV2          | 否           | 20      | TripletLoss  | [下载链接](https://pan.quark.cn/s/329d3c2dd2df) | 93.06%     |
| MobileNetV2          | 是           | 20      | TripletLoss   | [下载链接](https://pan.quark.cn/s/1503eb01677e) | 94.27%     |
| Inception_resnet_v1  | 否           | 20    | TripletLoss  | [下载链接](https://pan.quark.cn/s/deb09ac34965)                                    | 92.17%         |
| Inception_resnet_v1  | 是           | 20       | TripletLoss   | [下载链接](https://pan.quark.cn/s/07a4bbfd8bb9)                                    | 91.22%         |


## 配置文件解析
# 配置参数说明

## 训练配置参数

| 参数                        | 说明                                                         | 默认值                                      |
|-----------------------------|--------------------------------------------------------------|---------------------------------------------|
| **train.batch_size**         | 每个训练批次的样本数。                                         | 32                                          |
| **train.epoch**              | 训练的总轮数。                                                | 20                                          |
| **train.num_workers**        | 数据加载时使用的线程数。                                       | 8                                           |
| **train.device**             | 训练设备，`cuda` 或 `cpu`。                                   | `cuda`                                      |
| **train.momentum**           | 优化器的动量参数。                                            | 0.9                                         |
| **train.Init_lr**            | 初始学习率。                                                  | 0.001                                       |
| **train.optimizer**          | 选择优化器，可为 `sgd` 或 `adam`。                            | `sgd`                                       |
| **train.txt_file_path**      | 训练数据路径文件。                                             | `D:\code\dataset\webface\train.txt`         |
| **train.lr_decay_type**      | 学习率衰减类型，支持 `cos` 和 `step`。                        | `cos`                                       |
| **train.nbs**                | 基准批次大小，用于动态调整学习率。                             | 16                                          |
| **train.output**             | 模型输出目录。                                                | `output`                                    |
| **train.save_interval**      | 模型保存的间隔（按轮数）。                                    | 3                                           |
| **train.resume**             | 恢复训练的模型路径。                                          | `output\20241016_112734\epoch_12.pth`       |
| **train.model_data**         | 预训练模型的路径。                                            | `model_data\mobilenet_v2-b0353104.pth`      |
| **train.loss_type**          | 损失函数类型，支持 `ContrastLoss` 和 `TripletLoss`。           | `ContrastLoss`                              |

### 辅助分类器配置

| 参数                              | 说明                                                         | 默认值                                      |
|-----------------------------------|--------------------------------------------------------------|---------------------------------------------|
| **train.aux_classifier.use**      | 是否启用辅助分类器。                                           | `false`                                     |
| **train.aux_classifier.num_classes** | 辅助分类器的类别数。                                         | 10575                                       |

### LFW 数据集配置

| 参数                              | 说明                                                         | 默认值                                      |
|-----------------------------------|--------------------------------------------------------------|---------------------------------------------|
| **train.lfw.path**                | LFW 数据集的路径。                                            | `D:\code\dataset\lfw`                       |
| **train.lfw.pair_path**           | LFW 验证对文件路径。                                          | `./lfw_pair.txt`                            |

## 预测配置参数

| 参数                        | 说明                                                         | 默认值                                      |
|-----------------------------|--------------------------------------------------------------|---------------------------------------------|
| **predict.device**           | 预测时使用的设备。                                             | `cuda`                                      |
| **predict.mode**             | 预测模式，可为 `image` 或其他模式。                            | `image`                                     |
| **predict.image_path**       | 预测时的图像路径。                                             | `img/person.jpg`                            |
| **predict.model_path**       | 预测时的模型路径。                                             | `output\20241009_001559\best.pth`           |
| **predict.num_workers**      | 预测时使用的数据加载线程数。                                    | 4                                           |
| **predict.batch_size**       | 预测批次大小。                                                 | 4                                           |

## 输入尺寸配置

| 参数                        | 说明                                                         | 默认值                                      |
|-----------------------------|--------------------------------------------------------------|---------------------------------------------|
| **input_shape**              | 输入图像的尺寸，宽和高。                                      | `[128, 128]`                                |

## 评估配置参数

| 参数                        | 说明                                                         | 默认值                                      |
|-----------------------------|--------------------------------------------------------------|---------------------------------------------|
| **eval.dataset_path**        | 评估时的数据集路径。                                           | `D:\code\dataset\face\all`                  |
| **eval.device**              | 评估时使用的设备。                                             | `cuda`                                      |
| **eval.model_path**          | 评估时使用的模型路径。                                         | `output\20241012_233443\epoch_4.pth`        |
| **eval.batch_size**          | 评估时的批次大小。                                             | 16                                          |
| **eval.num_workers**         | 评估时的数据加载线程数。                                       | 8                                           |

## 主干网络和随机种子

| 参数                        | 说明                                                         | 默认值                                      |
|-----------------------------|--------------------------------------------------------------|---------------------------------------------|
| **backbone**                 | 主干网络，支持 `Inception_resnet_v1` 和 `mobilenet_v2`。       | `mobilenet_v2`                              |
| **seed**                     | 随机种子，用于控制实验的可重复性。                             | 42                                          |



## 数据集

本项目所使用的数据集来自于 [bubbliiiing
] 的开源项目 [facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch?tab=MIT-1-ov-file)。感谢原作者的贡献！

训练数据集使用经过了人脸矫正后的WebFace数据集
数据集链接：[WebFace+lfw]( https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw)。



## 参考

本项目参考了 [facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch?tab=MIT-1-ov-file)，该项目由 [bubbliiiing] 维护，遵循 MIT 许可证。

感谢 [bubbliiiing] 的贡献，其项目为本项目提供了重要的思路与实现参考。
