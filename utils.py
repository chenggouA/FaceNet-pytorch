
from tqdm import tqdm
import torch
from torch.utils.tensorboard.writer import SummaryWriter

def fit_one_epoch(writer: SummaryWriter, trainer, dataLoader, epoch, EPOCH, cuda, epoch_steps):
    print('Start Train')
    trainer.train()
    trainer.clear_loss()

    total_step = epoch_steps * epoch


    with tqdm(total=epoch_steps, desc=f'Epoch {epoch + 1} / {EPOCH}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataLoader):
            
            if iteration >= epoch_steps: break

            if cuda:
                # 如果 batch 是元组或列表，可以将每个元素都转移到 GPU 上
                batch = [item.cuda(non_blocking=True) for item in batch]


            losses = trainer.train_step(*batch)
            result_dict: dict = trainer.get_result_dict(losses)
            
            for k, v in result_dict.items():
                writer.add_scalar(f"loss/{k}", v / (iteration + 1), total_step + iteration)

            writer.add_scalar("lr", trainer.get_lr(), total_step + iteration)
            
            pbar.set_postfix(**{k: v / (iteration + 1) for k, v in result_dict.items()})
            pbar.update(1)
  