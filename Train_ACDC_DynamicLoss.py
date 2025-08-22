"""
ACDC数据集的测试函数 使用了动态Loss进行测试
"""

import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import tqdm
import os
import torch.nn.functional as F
import random
import math
import time

# 指定GPU训练,如果使用多GPU，下面这一行要注释掉
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
from torch.optim.lr_scheduler import StepLR # 动态学习率
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

# Synapse的dataloader
from DataLoader_Synapse import Synapse_dataset
from DataLoader_Synapse import RandomGenerator as Synapse_RandomGenerator

# ACDC的dataloader
from DataLoader_ACDC import ACDCdataset as ACDC_dataset
from DataLoader_ACDC import RandomGenerator as ACDC_RandomGenerator

# CVCDB的Dataloader
# from DataLoader.DataLoader_CVCClinicDB import CVCClinicDB_Dataset as CVCDB_dataset
# from DataLoader.DataLoader_CVCClinicDB import ExtCompose
# from DataLoader.DataLoader_CVCClinicDB import ExtResize
# from DataLoader.DataLoader_CVCClinicDB import ExtRandomRotation
# from DataLoader.DataLoader_CVCClinicDB import ExtRandomHorizontalFlip
# from DataLoader.DataLoader_CVCClinicDB import ExtToTensor

# 导入网络框架
# from network.LKA_Source import UNet # 导入LKA_Source网络
# from network.DDL_Source import UNet # 导入DDL_Source网络
# from network.DDL_Source_2Layer import UNet # 导入两层的网络进行尝试
# from network.DDL_Source_3Layer import UNet # 导入三层网络进行尝试
# from network.DDL_Source_64channel import UNet # 导入64层通道进行尝试    
from network.DDL_Source_128channel import UNet # 导入128层的网络进行尝试
# from network.DDL_Rewrite import Network

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

# 只能用于二分类
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

def bceloss_test():
    """BCELoss Test函数"""
    pred = torch.sigmoid(torch.randn(3, 1, 224, 224))  # 模拟预测值，使用 sigmoid 激活函数，把其限制在[0,1]范围内
    target = torch.randint(0, 2, (3, 1, 224, 224)).float()  # 模拟目标值，二分类问题
    print(f'predict shape is {pred.shape}')
    print(f'target shape is {target.shape}')

    # 初始化 BCELoss
    loss_fn = BCELoss()

    # 计算损失
    loss = loss_fn(pred, target)

    # 打印损失
    print(f"BCELoss: {loss.item()}")

class DiceLoss(nn.Module):
    """Dice Loss函数"""
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1 # 防止分母为0
        size = pred.size(0) # Batchsize大小

        # 将预测值和目标值展开成一维度
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        # 计算交集
        intersection = pred_ * target_

        # 计算Dice系数
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        
        # 计算DiceLoss
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

def diceloss_test():
    """DiceLoss函数测试"""
    pred = torch.sigmoid(torch.randn(3, 9, 224, 224))  # 模拟预测值，使用 sigmoid 激活函数
    target = torch.randint(0, 2, (3, 9, 224, 224)).float()  # 模拟目标值，二分类问题

    print(f'predict shape is {pred.shape}')
    print(f'target shape is {target.shape}')

    # 初始化 DiceLoss
    loss_fn = DiceLoss()

    # 计算损失
    loss = loss_fn(pred, target)

    # 打印损失
    print(f"DiceLoss: {loss.item()}")


class nDiceLoss(nn.Module):
    """多类别DiceLoss"""
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    # 将输入的目标标签进行 one-hot 编码，生成 [batch, n_classes, ...] 的张量。
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []

        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    # 计算单个类别的Loss
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    # 向前传播
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

def ndiceloss_test():
    """多类别DiceLoss函数进行测试"""
    pred = torch.randn(3, 9, 224, 224)  # 模拟预测值
    target = torch.randint(0, 9, (3, 224, 224)).long()  # 模拟目标值，取值范围为[0, 8]

    print(f'predict shape is {pred.shape}')
    print(f'target shape is {target.shape}')

    # 初始化 nDiceLoss
    loss_fn = nDiceLoss(n_classes=9)

    # 计算损失
    loss = loss_fn(pred, target, softmax=True)

    # 打印损失
    print(f"nDiceLoss: {loss.item()}")

class CeDiceLoss(nn.Module):
    """固定权重的CeDiceLoss"""
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
    """BCEloss和DiceLoss的相互结合"""
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

# 更新的动态Loss
class CeDiceLoss_functionnal(nn.Module):
    """动态权重的CeDiceLoss"""

    def __init__(self, num_classes, initial_loss_weight=[1, 1]):
        super(CeDiceLoss_functionnal, self).__init__()

        self.celoss = nn.CrossEntropyLoss()  # 设置交叉熵损失
        self.diceloss = nDiceLoss(num_classes)  # 设置DiceLoss
        self.loss_weight = initial_loss_weight  # 设置初始化权重
        self.num_classes = num_classes  # 设置分割多分类数量

    def forward(self, pred, target, epoch=None):
        """
        Input: predicted image, target image, present epoch number
        Output: loss,loss_ce,loss_dice
        """

        if epoch is not None:
            self.loss_weight = self._adjust_loss_weights_2(epoch)

        loss_ce = self.celoss(pred, target.long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss, loss_ce, loss_dice

    def _adjust_loss_weights_1(self, epoch, k=0.1, x0=100):
        """动态调整权重的策略"""
        sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - x0))) # k为斜率，x0为中点偏移值，一般设置成epoch的一半
        
        # 动态调整权重，使得a从1逐渐变到0，b从1逐渐变到2
        a = 1 - sigmoid(epoch)
        b = 1 + sigmoid(epoch)
        
        return [a, b]
    
    def _adjust_loss_weights_2(self, epoch, k=0.1, x0=100):
        """动态调整权重的策略第二种"""
        sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - x0)))  # k为斜率，x0为中点偏移值，一般设置成epoch的一半

        # 动态调整权重，使得a从1逐渐变到0.6，b从1逐渐变到1.4
        a = 0.6 + (1 - 0.6) * (1 - sigmoid(epoch))  # 从1到0.6
        b = 1 + (1.4 - 1) * sigmoid(epoch)  # 从1到1.4
        
        return [a, b]
    
    def _adjust_loss_weights_3(self, epoch, k=0.1, x0=150):
        """动态调整权重的策略的第三种，适用于300Epoch"""
        sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - x0))) # k为斜率，x0为中点偏移值，一般设置成epoch的一半
        
        # 动态调整权重，使得a从1逐渐变到0，b从1逐渐变到2
        a = 1 - sigmoid(epoch)
        b = 1 + sigmoid(epoch)
        
        return [a, b]


# 测试 CeDiceLoss_functionnal
def cedice_loss_functionally_test():
    # 创建模拟输入
    pred = torch.randn(3, 9, 224, 224)  # 模拟预测值
    target = torch.randint(0, 9, (3, 224, 224)).long()  # 模拟目标值，取值范围为[0, 8]

    # 初始化 CeDiceLoss_functionnal
    loss_fn = CeDiceLoss_functionnal(num_classes=9)

    # 计算损失
    for epoch in range(0, 200, 10):
        total_loss, loss_ce, loss_dice = loss_fn(pred, target, epoch=epoch)
        print(f"Epoch {epoch}, Total Loss: {total_loss.item()}, Cross Entropy Loss: {loss_ce.item()}, Dice Loss: {loss_dice.item()}")
        print(f"Loss Weights: {loss_fn.loss_weight}")

class BCEWithLogitsLoss(nn.Module):
    """使用了log函数控制了区间的BCELoss"""
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
    """似乎是另外一种BceDiceLoss的变种"""
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
    
    # 设置随机种子
    set_seed(42)

    # 配置日志系统，添加了覆盖功能
    if continue_train and os.path.exists(train_log_path):
        log_dir = train_log_path
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(train_log_path, timestamp)

    # 设计Tensorboard地址
    writer = SummaryWriter(log_dir=log_dir)
    
    # 使用 DataLoader 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 配置优化器
    logging.info(f"Optimizer: {optimizer.__class__.__name__} with parameters: {optimizer.defaults}")

    # 设置Loss函数
    criterion = CeDiceLoss_functionnal(num_classes=4,initial_loss_weight=[1,1]) # 用于动态Loss
    # criterion = CeDiceLoss(num_classes=9,loss_weight=[0.6,1.4])
    # criterion = CeDiceLoss(num_classes=4,loss_weight=[1,1])
    # criterion = CeDiceLoss(num_classes=4,loss_weight=[0.6,1.4])
    # criterion = CeDiceLoss(num_classes=4,loss_weight=[1,1])
    logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: now using dynamic loss function,start with [1,1],end with[0.6,1.4]")
    # logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: [1,1]")
    # logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: [0.6,1.4]")

    # 设置学习计划
    scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.9)  # 动态学习率
    logging.info(f"Scheduler: {scheduler.__class__.__name__} with step_size={scheduler.step_size}, gamma={scheduler.gamma}")

    # 将模型设置为训练模式
    model.train()
   
    
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
        running_loss_ce = 0.0
        running_loss_dice = 0.0
        
        pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for i_batch, sampled_batch in enumerate(train_loader):
            
            # 判断 sampled_batch 的类型
            if isinstance(sampled_batch, dict):
                images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            else:
                images, labels = sampled_batch
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 从损失函数得到三个返回值 动态Loss会返回三个数值
            loss, loss_ce, loss_dice = criterion(outputs, labels)
            # loss = criterion(outputs,labels)

            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_dice += loss_dice.item()
            
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i_batch)
            writer.add_scalar('Loss_CE/train', loss_ce.item(), epoch * len(train_loader) + i_batch)
            writer.add_scalar('Loss_Dice/train', loss_dice.item(), epoch * len(train_loader) + i_batch)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(i_batch+1):.4f}',
                'Loss_CE': f'{running_loss_ce/(i_batch+1):.4f}',
                'Loss_Dice': f'{running_loss_dice/(i_batch+1):.4f}',
                'LR': scheduler.get_last_lr()[0]
            })
            pbar.update(1)
        
        pbar.close()
        
        # 计算并打印每个 epoch 的平均损失
        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.4f}, CE Loss: {loss_ce:.4f}, Dice Loss: {loss_dice:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")
        # logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")

        # 记录平均损失和学习率
        writer.add_scalar("Average Loss/train", epoch_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # 更新学习率
        scheduler.step()

        # 每隔一定的 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            temp_path = os.path.join(save_path, f'model_epoch_{epoch+1}_checkpoint.pth')
            
            if multi_gpu:
                torch.save(model.module.state_dict(), temp_path)  # 多卡模型用这个
            else:
                torch.save(model.state_dict(), temp_path)  # 单卡模型用这个

            logging.info(f"Saved checkpoint at epoch {epoch+1} at {temp_path}")

    writer.close()
    logging.info("Training Complete!")

def old_main():
    pass

if __name__ == "__main__":
    # 参数设计
    parser = argparse.ArgumentParser(description="Train a deep learning model on a given dataset")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--log_path", type=str, default="./result/Training.log", help="Path to save running log")
    parser.add_argument("--img_size", type=int, default=224, help="The image size for train, usually 224 or 256 or 512")
    parser.add_argument("--pth_path", type=str, default='./result/Pth', help="The path to save running pth")
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train', help="The path to save tensorboard file")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from latest checkpoint")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs to train the model")
    parser.add_argument("--dataset", choices=['synapse', 'acdc', 'cvcdb'], default='synapse', help="Dataset to use: 'synapse' or 'acdc' or 'cvcdb'")
    option = parser.parse_args()    

    
    # 设置日志位置
    setup_logging(option.log_path)

    # 记录运行的参数
    logging.info(f"Running with parameters: {vars(option)}")

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Now is Going to use {device.type}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    # 选择不同的训练数据集
    if option.dataset == 'synapse':
        transform = Synapse_RandomGenerator((option.img_size, option.img_size))
        db_train = Synapse_dataset(
            base_dir="/samba/network-storage/ssd/code/datasets/Synapse/data",
            list_dir="/samba/network-storage/ssd/code/datasets/Synapse/list",
            split="train",
            transform=transform
        )
    elif option.dataset == 'acdc':
        transform = ACDC_RandomGenerator((option.img_size, option.img_size))
        db_train = ACDC_dataset(
            base_dir="/samba/network-storage/ssd/code/datasets/ACDC",  
            list_dir="/samba/network-storage/ssd/code/datasets/ACDC/lists_ACDC", 
            split="train",
            transform=transform
        )
    # elif option.dataset == 'cvcdb':
    #     transform = ExtCompose([
    #     ExtResize((option.img_size, option.img_size)),
    #     ExtRandomRotation(degrees=90),
    #     ExtRandomHorizontalFlip(),
    #     ExtToTensor()
    #     ])
    #     db_train = CVCDB_dataset(
    #         root="/samba/network-storage/ssd/code/datasets/CVC-ClinicDB/",
    #         dataset_type="train",
    #         cross="1",
    #         transform=transform
    #     )
    else:
        print(f'you input a error dataset! please check you code in Train.py!')

    # 模型实例化
    model = UNet(n_channels=1,n_classes=4).to(device=device)
    # model = Network(in_channel=1,out_channel=96,final_channel=4).to(device=device)

    # 加载最新的模型继续训练
    if option.continue_train:
        checkpoint = latest_checkpoint(option.pth_path)
        if checkpoint:
            load_model(model=model, model_path=checkpoint, device=device)  # 自动处理单卡 pth 和多卡 pth 文件
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
                continue_train=option.continue_train, multi_gpu=option.multi_gpu)