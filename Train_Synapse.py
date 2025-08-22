"""
Synapse的训练代码，使用的时阶段性损失函数
"""
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import tqdm
import os
import torch.nn.functional as F
import random

# 指定GPU训练,如果使用多GPU，下面这一行要注释掉
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
from torch.optim.lr_scheduler import StepLR # 动态学习率
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

from DataLoader_Synapse import Synapse_dataset
from DataLoader_Synapse import RandomGenerator

# 导入网络框架
# from network.MissFormer.MISSFormer import MISSFormer # 导入MissFormer网络
from network.Unet2D import UNet

def set_seed(seed_value=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file):
    """日志记录"""
    log_dir = os.path.dirname(log_file)
    # 如果日志文件夹没有，就创建文件夹
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])
    
    # 测试日志记录器是否正常工作
    logging.info("Logging is set up.")

# 查找最新的文件
def latest_checkpoint(path):
    """在path中查找出最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

class BCELoss(nn.Module):
    """解决了内存不连续的问题"""
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        # 使用reshape代替view
        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)

        return self.bceloss(pred_, target_)

class DiceLoss(nn.Module):
    """Dice Loss函数"""
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    
class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

# 使用动态权重的CeDiceLoss
class CeDiceLoss_Dynamic(nn.Module):
    # 设置初始CE权重和初始Dice权重
    def __init__(self, num_classes, initial_ce_weight=1, initial_dice_weight=1):
        super(CeDiceLoss_Dynamic, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = initial_ce_weight
        self.dice_weight = initial_dice_weight

    def forward(self, outputs, targets, epoch, total_epochs):
        ce_loss = F.cross_entropy(outputs, targets)
        dice_loss = self.dice_loss(outputs, targets)
        
        # 动态调整权重
        ce_weight = self.ce_weight * (1 - epoch / total_epochs)
        dice_weight = self.dice_weight * (epoch / total_epochs)
        
        loss = ce_weight * ce_loss + dice_weight * dice_loss
        return loss

    def dice_loss(self, outputs, targets):
        smooth = 1.0
        outputs = torch.softmax(outputs, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (outputs * one_hot_targets).sum(dim=(2, 3))
        union = outputs.sum(dim=(2, 3)) + one_hot_targets.sum(dim=(2, 3))

        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        return dice_loss.mean()

# 无需使用log函数
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(torch.sigmoid(pred), target)  # Apply sigmoid here for DiceLoss

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss

# 自动处理多卡和单卡模型
def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # If loading a DataParallel model, remove `module.` prefix
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # remove `module.`
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model

# 训练函数
def train_model(model, train_dataset, epochs=300, batch_size=1, learning_rate=1e-4,
                save_path=None, train_log_path=None, continue_train=None, multi_gpu=False):
    # 配置日志系统
    if continue_train and os.path.exists(train_log_path):
        writer = SummaryWriter(log_dir=train_log_path, purge_step=None)
    else:
        writer = SummaryWriter(log_dir=train_log_path)
    
    # 使用 DataLoader 加载数据集
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # 新版本的dataloader

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)  # 配置优化器
    logging.info(f"Optimizer: {optimizer.__class__.__name__} with parameters: {optimizer.defaults}")

    # 设置Loss函数
    # criterion = CeDiceLoss(num_classes=9, loss_weight=[0.6, 1.4])
    criterion = CeDiceLoss(num_classes=9,loss_weight=[1,1])
    # criterion = CeDiceLoss(num_classes=9,loss_weight=[0.4,1.6])
    # logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: num_classes=9, loss_weight=[0.6, 1.4]")
    # logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: num_classes=9, loss_weight=[1, 1]")
    logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: num_classes=9, loss_weight=[0.4, 1.6]")

    # 设置学习计划
    scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.9)  # 动态学习率
    logging.info(f"Scheduler: {scheduler.__class__.__name__} with step_size={scheduler.step_size}, gamma={scheduler.gamma}")


    # 将模型设置为训练模式
    model.train()

    # 设置随机种子
    set_seed(42)
    
    # 设置多卡和device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if multi_gpu:
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")

    # tqdm训练过程
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i_batch)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'Loss': f'{running_loss/(i_batch+1):.4f}', 'LR': scheduler.get_last_lr()[0]})
            pbar.update(1)
        
        pbar.close()
        
        # 计算并打印每个 epoch 的平均损失
        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")

        # 记录平均损失和学习率
        writer.add_scalar("Average Loss/train", epoch_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # 更新学习率
        scheduler.step()

        # 每隔一定的 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            temp_path = os.path.join(save_path, f'model_epoch_{epoch+1}_checkpoint.pth')
            
            torch.save(model.state_dict(), temp_path)  # 单卡模型用这个

            logging.info(f"Saved checkpoint at epoch {epoch+1} at {temp_path}")

    writer.close()
    logging.info("Training Complete!")


if __name__ == "__main__":

    # 参数设计
    parser = argparse.ArgumentParser(description="Train a deep learning model on a given dataset")
    parser.add_argument("--epochs",type=int,default=300,help="Number of epochs to train")
    parser.add_argument("--batch_size",type=int,default=12,help="Batch size for training")
    parser.add_argument("--learning_rate",type=float,default=1e-5,help="Inital learning rate")
    parser.add_argument("--log_path",type=str,default="./result/running.log",help="path to save running log")
    parser.add_argument("--img_size",type=int,default=224,help="the image size for train,usually 224 or 256 or 512")
    parser.add_argument("--pth_path",type=str,default='./result/Pth',help="the path for save running pth")
    parser.add_argument("--tensorboard_path",type=str,default='./result/Train',help="the path for save tensorboard file")
    parser.add_argument("--continue_train",action="store_true",help="Continue training from latest checkpoint")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs to train the model")

    option = parser.parse_args()

    # 设置日志位置
    setup_logging(option.log_path)

    # 记录运行的参数
    logging.info(f"Running with parameters: {vars(option)}")

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Now is Going to use {device.type}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # 数据集和模型的初始化 transform = RandomGenerator((224, 224))
    db_train = Synapse_dataset(base_dir="./datasets/Synapse/data", list_dir="./datasets/Synapse/list", split="train",transform = RandomGenerator((option.img_size,option.img_size)))

    # 模型实例初始化
    model = UNet(n_channels=1, n_classes=9).to(device)
    # model = Network(in_channel=1,out_channel=96,final_channel=9).to(device=device) # DDL_Rewrite
    # model = ULite().to(device=device)
    # model = MISSFormer(num_classes=9).to(device=device)
    # model = AttentionUNet(img_ch=1,output_ch=9).to(device=device)

    # 加载最新的模型继续训练
    if option.continue_train:
        checkpoint = latest_checkpoint(option.pth_path)
        if checkpoint:
            # model.load_state_dict(torch.load(checkpoint, map_location=device))
            load_model(model=model,model_path=checkpoint,device=device) # 自动处理单卡pth和多卡pth文件
            logging.info(f"Continuing training from {checkpoint}")
        else:
            logging.info("No checkpoint found, starting a new training session")
        
    # 确保保存模型的目录存在
    if not os.path.exists(option.pth_path):
        os.makedirs(option.pth_path)

    if not os.path.exists(option.tensorboard_path):
        os.makedirs(option.tensorboard_path)
    
    # 开始训练
    train_model(model, db_train, epochs=option.epochs, batch_size=option.batch_size, learning_rate=option.learning_rate, save_path=option.pth_path, train_log_path=option.tensorboard_path,
                continue_train=option.continue_train,multi_gpu=option.multi_gpu)
