"""
ACDC Training Code
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

import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
from glob import glob

from DataLoader_Synapse import Synapse_dataset
from DataLoader_Synapse import RandomGenerator as Synapse_RandomGenerator
from DataLoader_ACDC import ACDCdataset as ACDC_dataset
from DataLoader_ACDC import RandomGenerator as ACDC_RandomGenerator

from network.DLKUNet import UNet

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])
    
    logging.info("Logging is set up.")

def latest_checkpoint(path):
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)

        return self.bceloss(pred_, target_)

def bceloss_test():
    pred = torch.sigmoid(torch.randn(3, 1, 224, 224)) 
    target = torch.randint(0, 2, (3, 1, 224, 224)).float()
    print(f'predict shape is {pred.shape}')
    print(f'target shape is {target.shape}')

    loss_fn = BCELoss()

    loss = loss_fn(pred, target)

    print(f"BCELoss: {loss.item()}")

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

def diceloss_test():
    """DiceLoss函数测试"""
    pred = torch.sigmoid(torch.randn(3, 9, 224, 224))
    target = torch.randint(0, 2, (3, 9, 224, 224)).float()

    print(f'predict shape is {pred.shape}')
    print(f'target shape is {target.shape}')

    loss_fn = DiceLoss()

    loss = loss_fn(pred, target)

    print(f"DiceLoss: {loss.item()}")


class nDiceLoss(nn.Module):
    """多类别DiceLoss"""
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []

        for i in range(self.n_classes):
            temp_prob = input_tensor == i
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

def ndiceloss_test():
    pred = torch.randn(3, 9, 224, 224)
    target = torch.randint(0, 9, (3, 224, 224)).long()

    print(f'predict shape is {pred.shape}')
    print(f'target shape is {target.shape}')

    loss_fn = nDiceLoss(n_classes=9)

    loss = loss_fn(pred, target, softmax=True)

    print(f"nDiceLoss: {loss.item()}")

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

class CeDiceLoss_functionnal(nn.Module):

    def __init__(self, num_classes, initial_loss_weight=[1, 1]):
        super(CeDiceLoss_functionnal, self).__init__()

        self.celoss = nn.CrossEntropyLoss()  
        self.diceloss = nDiceLoss(num_classes)  
        self.loss_weight = initial_loss_weight  
        self.num_classes = num_classes  

    def forward(self, pred, target, epoch=None):
        """
        Input: predicted image, target image, present epoch number
        Output: loss,loss_ce,loss_dice
        """

        if epoch is not None:
            self.loss_weight = self._adjust_loss_weights_3(epoch)

        loss_ce = self.celoss(pred, target.long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss, loss_ce, loss_dice

    def _adjust_loss_weights_1(self, epoch, k=0.1, x0=100):
        sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - x0))) 
        a = 1 - sigmoid(epoch)
        b = 1 + sigmoid(epoch)
        
        return [a, b]
    
    def _adjust_loss_weights_2(self, epoch, k=0.1, x0=100):
        sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - x0))) 
        a = 0.6 + (1 - 0.6) * (1 - sigmoid(epoch))
        b = 1 + (1.4 - 1) * sigmoid(epoch)
        
        return [a, b]
    
    def _adjust_loss_weights_3(self, epoch, k=0.1, x0=150):
        sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - x0)))
        a = 1 - sigmoid(epoch)
        b = 1 + sigmoid(epoch)
        
        return [a, b]

def cedice_loss_functionally_test():
    pred = torch.randn(3, 9, 224, 224) 
    target = torch.randint(0, 9, (3, 224, 224)).long() 
    loss_fn = CeDiceLoss_functionnal(num_classes=9)

    for epoch in range(0, 200, 10):
        total_loss, loss_ce, loss_dice = loss_fn(pred, target, epoch=epoch)
        print(f"Epoch {epoch}, Total Loss: {total_loss.item()}, Cross Entropy Loss: {loss_ce.item()}, Dice Loss: {loss_dice.item()}")
        print(f"Loss Weights: {loss_fn.loss_weight}")

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(torch.sigmoid(pred), target)  

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

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model

def train_model(model, train_dataset, epochs=300, batch_size=1, learning_rate=1e-4,
                save_path=None, train_log_path=None, continue_train=None, multi_gpu=False):
    
    set_seed(42)

    if continue_train and os.path.exists(train_log_path):
        log_dir = train_log_path
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(train_log_path, timestamp)

    writer = SummaryWriter(log_dir=log_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logging.info(f"Optimizer: {optimizer.__class__.__name__} with parameters: {optimizer.defaults}")
    criterion = CeDiceLoss(num_classes=4,loss_weight=[0.4,0.6])
    logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: [0.4,0.6]")

    scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
    logging.info(f"Scheduler: {scheduler.__class__.__name__} with step_size={scheduler.step_size}, gamma={scheduler.gamma}")

    model.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if multi_gpu:
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")

    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_dice = 0.0
        
        pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for i_batch, sampled_batch in enumerate(train_loader):
            
            if isinstance(sampled_batch, dict):
                images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            else:
                images, labels = sampled_batch
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs,labels)

            running_loss += loss.item()
            
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i_batch)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(i_batch+1):.4f}',
                'LR': scheduler.get_last_lr()[0]
            })
            pbar.update(1)
        
        pbar.close()

        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")

        writer.add_scalar("Average Loss/train", epoch_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            temp_path = os.path.join(save_path, f'model_epoch_{epoch+1}_checkpoint.pth')
            
            if multi_gpu:
                torch.save(model.module.state_dict(), temp_path)  
            else:
                torch.save(model.state_dict(), temp_path)

            logging.info(f"Saved checkpoint at epoch {epoch+1} at {temp_path}")

    writer.close()
    logging.info("Training Complete!")

if __name__ == "__main__":

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

    setup_logging(option.log_path)

    logging.info(f"Running with parameters: {vars(option)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Now is Going to use {device.type}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    if option.dataset == 'synapse':
        transform = Synapse_RandomGenerator((option.img_size, option.img_size))
        db_train = Synapse_dataset(
            base_dir="./datasets/Synapse/data",
            list_dir="./datasets/Synapse/list",
            split="train",
            transform=transform
        )
    elif option.dataset == 'acdc':
        transform = ACDC_RandomGenerator((option.img_size, option.img_size))
        db_train = ACDC_dataset(
            base_dir="./datasets/ACDC",  
            list_dir="./datasets/ACDC/lists_ACDC", 
            split="train",
            transform=transform
        )
    else:
        print(f'you input a error dataset! please check you code in Train.py!')

    model = UNet(n_channels=1,n_classes=4).to(device=device)

    if option.continue_train:
        checkpoint = latest_checkpoint(option.pth_path)
        if checkpoint:
            load_model(model=model, model_path=checkpoint, device=device)  
            logging.info(f"Continuing training from {checkpoint}")
        else:
            logging.info("No checkpoint found, starting a new training session")
        
    if not os.path.exists(option.pth_path):
        os.makedirs(option.pth_path)

    if not os.path.exists(option.tensorboard_path):
        os.makedirs(option.tensorboard_path)
    
    train_model(model, db_train, epochs=option.epochs, batch_size=option.batch_size, learning_rate=option.learning_rate, save_path=option.pth_path, train_log_path=option.tensorboard_path,
                continue_train=option.continue_train, multi_gpu=option.multi_gpu)