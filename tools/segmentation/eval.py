import numpy as np
import torch 
from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # Return overall pixel accuracy
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # Return each category pixel accuracy
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc
 
    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def diceCoefficient(self):
        # Compute Dice Coefficient for each class
        TP = np.diag(self.confusionMatrix)
        FP = np.sum(self.confusionMatrix, axis=0) - TP
        FN = np.sum(self.confusionMatrix, axis=1) - TP
        
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)  # Adding epsilon to avoid division by zero
        return dice
 
    def meanDiceCoefficient(self):
        # Compute mean Dice Coefficient
        dice = self.diceCoefficient()
        mean_dice = np.nanmean(dice)
        return mean_dice

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def val_one_epoch(num_classes, writer: SummaryWriter, trainer, test_dataLoader, epoch, EPOCH):
    
    metric = SegmentationMetric(num_classes)
    val_total_loss = 0.0
    print("Start Test")
    trainer.eval()
    
    with tqdm(total=len(test_dataLoader), desc=f'Epoch {epoch + 1} / {EPOCH}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(test_dataLoader):
            with torch.no_grad():
                images, pngs, seg_labels = batch[0], batch[1], batch[2]
                
                outputs: Tensor
                loss, outputs = trainer(images, pngs, seg_labels)

                val_total_loss  += loss.item()


                outputs = outputs.argmax(dim=1).cpu().detach().numpy().reshape(-1)

                pngs = pngs.cpu().numpy().reshape(-1)

                metric.addBatch(outputs, pngs)
                
                del images, outputs, pngs

                
            pbar.set_postfix(**{'total_loss': val_total_loss / (iteration + 1)})
            pbar.update(1)


    # 释放不需要的变量并清理缓存
    torch.cuda.empty_cache()
    
    
    writer.add_scalar("dice_per_class", metric.meanDiceCoefficient(), epoch)
    writer.add_scalar("mPA", metric.meanPixelAccuracy(), epoch)
    writer.add_scalar("mIou", metric.meanIntersectionOverUnion(), epoch)
    writer.add_scalar("loss/val", val_total_loss / len(test_dataLoader), epoch) 

    return val_total_loss / len(test_dataLoader)