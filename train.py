import torch.utils
import torch.utils.data
from tools.config import Config, load_config
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from tools.train import set_seed
from tools.train import EarlyStopping
from tools.sys import create_folder_with_current_time
import os
from utils import fit_one_epoch
from trainer import Trainer
from loss import ComputeLoss
from dataset import LFWDataset, ContrastLossDataset, TripletLossDataset, FaceNet_dataset_collate
import torch
from facenet import FaceNet 
from tools.lr_scheduler import YOLOXCosineLR
from lfw import eval

torch.backends.cudnn.benchmark = True



def train(config: Config, output, writer: SummaryWriter, train_dataLoader, lfw_dataLoader, EPOCH, Init_lr_fit, Min_lr_fit, epoch_steps):
   
    save_interval = config['train.save_interval']
    resume = config['train.resume']
    device = config['train.device']
    optimizer_type = config['train.optimizer']
    momentum = config['train.momentum']
    num_classes = config['train.aux_classifier.num_classes']
    start_epoch = 0
    aux_classifier = config['train.aux_classifier.use']
    loss_type = config['train.loss_type']

    
    cuda = False
    if device == "cuda":
        cuda = True
    

    
    
    def get_optimizer(model):
        params = [p for p in model.parameters() if p.requires_grad ]
        optimizer = {
                    'adam'  : optim.Adam(params, Init_lr_fit, betas = (momentum, 0.999), weight_decay = 0),
                    'sgd'   : optim.SGD(params, Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = 0)
        }[optimizer_type]

        return optimizer


    # 早停策略
    earlyStopping = EarlyStopping(output, save_interval, EPOCH, verbose=True)
    model = FaceNet(config['backbone'], aux_classifier=aux_classifier, num_classes=num_classes)
    model = model.to(device)
    trainer = Trainer(device, model, ComputeLoss(aux_classifier, num_classes, loss_type), aux_classifier)
    
    # 获得优化器
    optimizer = get_optimizer(model)

    #   获得学习率下降的公式
    lr_scheduler = YOLOXCosineLR(optimizer, Init_lr_fit, Min_lr_fit, epoch_steps * EPOCH)
    
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

        # 如果需要恢复训练，判断是否已经进入解冻阶段
        start_epoch = checkpoint['epoch'] + 1
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if device == "cuda":
            # 增加以下几行代码，将optimizer里的tensor数据全部转到GPU上
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()


        

    trainer.set_optimizer_and_lr_scheduler(optimizer, lr_scheduler)
    
    for epoch in range(start_epoch, EPOCH):
        
        fit_one_epoch(writer, trainer, train_dataLoader, epoch, EPOCH, cuda, epoch_steps)
        acc = eval(trainer, lfw_dataLoader, device, writer, epoch)
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_scheduler': lr_scheduler.state_dict()
        }
        
        earlyStopping(-acc, save_files, epoch)
        if earlyStopping.early_stop:
            print(f"[{epoch}/{EPOCH}], acc={acc} Stop!!!")
            break
    writer.close()

def main(config: Config):


    EPOCH = config['train.epoch']
    Init_lr = config['train.Init_lr']
    optimizer_type = config['train.optimizer']


    seed = config['seed']
    txt_file_path = config['train.txt_file_path']
    batch_size = config['train.batch_size']
    num_workers = config['train.num_workers']
    output = config['train.output']
    input_shape = config['input_shape']
    lfw_path = config['train.lfw.path']
    pair_path = config['train.lfw.pair_path']
    loss_type = config['train.loss_type']
    
    output = create_folder_with_current_time(output)


    set_seed(seed)

    

    # 初始化日志
    writer = SummaryWriter(log_dir=os.path.join(output, "log"))

    
    Min_lr = Init_lr * 0.01
 
    nbs             = config['train.nbs']
    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 获取数据集
    
    if loss_type == "ContrastLoss":
        train_dataset = ContrastLossDataset(txt_file_path, input_shape)
    elif loss_type == "TripletLoss":
        train_dataset = TripletLossDataset(txt_file_path, input_shape)

    lfw_dataset = LFWDataset(lfw_path, pair_path, input_shape)
        
    train_dataLoader = DataLoader(train_dataset, 
                                batch_size = batch_size, 
                                num_workers = num_workers, 
                                pin_memory = True,
                                collate_fn = FaceNet_dataset_collate,
                                drop_last = True)
    
    
    lfw_dataloader = torch.utils.data.DataLoader(
        lfw_dataset, 
        batch_size,
        drop_last=True
        )

    epoch_steps = len(train_dataset) // (batch_size * 3) if loss_type == "TripletLoss" else len(train_dataset) // (batch_size * 2)

    
    train(config, output, writer, train_dataLoader, lfw_dataloader, EPOCH, Init_lr_fit, Min_lr_fit, epoch_steps)
    

if __name__ == "__main__":
    main(load_config("application.yaml"))

