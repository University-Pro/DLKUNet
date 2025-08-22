"""
Synapse的测试代码
不保存测试的图片
"""
import numpy as np
import torch
from medpy import metric
import logging
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import os
import random

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

# 导入Dataloader
from DataLoader_Synapse import Synapse_dataset
from DataLoader_Synapse import RandomGenerator

# 导入网络
# from network.MissFormer.MISSFormer import MISSFormer
# from network.ULite import ULite
# from network.DDL_Rewrite import Network
# from network.DDL_Source import UNet
from network.Unet2D import UNet
# from network.DDL_Source_3Layer import UNet
# from network.LKA_Source import UNet
# from network.DDL_Source_64channel import UNet

def set_seed(seed_value=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = False
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

def latest_checkpoint(path):
    """查找path中最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# DiceLoss计算
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
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

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[512, 512], test_save_path=None, case=None, z_spacing=1):
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 读取图像内容并传输到GPU
    image, label = image.squeeze(0).to(device), label.squeeze(0).to(device)
    
    # 如果是3维
    if len(image.shape) == 3:
        prediction = torch.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = F.interpolate(slice.unsqueeze(0).unsqueeze(0), size=patch_size, mode='bilinear', align_corners=False).squeeze()
            input = slice.unsqueeze(0).unsqueeze(0).float().to(device)
            
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                if x != patch_size[0] or y != patch_size[1]:
                    pred = F.interpolate(out.unsqueeze(0).unsqueeze(0).float(), size=(x, y), mode='nearest').squeeze()
                else:
                    pred = out
                prediction[ind] = pred
    
    else:
        input = image.unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out
    
    # 将预测结果从GPU传输回CPU
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.cpu().numpy().astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

    return metric_list

# 模型加载设置，既支持多卡也支持单卡
def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    
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

# 接口
def inference(model, test_save_path=None, log_path=None, patch_size=None):
    logging.info("Starting inference process")

    # 测试的时候不需要进行打乱，num_workers最好设置成1
    db_test = Synapse_dataset(base_dir="./datasets/Synapse/data", list_dir="./datasets/Synapse/list", split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    
    # 模型进入评估模式，不进行梯度传播
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=9, patch_size=[patch_size, patch_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(db_test)

    for i in range(1, 9):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info('Testing Finished')

if __name__=='__main__':
    # 参数设计
    parser = argparse.ArgumentParser(description='Test a deep learning model on a given dataset and savecheckpoint')
    parser.add_argument("--model_load",type=str,default='./result/Pth/model_epoch_10_checkpoint.pth',help='The path of the checkpoint')
    parser.add_argument("--log_path",type=str,default='./result/Test/running.log',help='The path of the log')
    parser.add_argument("--patch_size",type=int,default=224,help='The resolution of test image size')
    parser.add_argument("--test_save_path",type=str,default=None,help="The path to save segmentation results")
    
    option = parser.parse_args()
    
    # 设定日志位置
    setup_logging(option.log_path)
    logging.info(f"log file will be setted at {option.log_path}")

    # 记录运行的参数
    logging.info(f"Running with parameters: {vars(option)}")

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Now is Going to use {device.type}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # 创建模型实例
    # model = MISSFormer(num_classes=9).to(device=device)
    model = UNet(n_channels=1, n_classes=9).to(device) # UNetLKA4 / UNetLKA5
    # model = ULite().to(device=device)
    # model = Network(in_channel=1,out_channel=96,final_channel=9).to(device=device)

    # 模型加载到设备上
    # model.load_state_dict(torch.load(option.model_load, map_location=device))
    model = load_model(model, option.model_load, device) # 新版本,自动处理单卡和多卡模型
    logging.info(f"Model loaded from {option.model_load}")

    # 调用你的测试函数
    inference(model, test_save_path=option.test_save_path,log_path=option.log_path,patch_size=option.patch_size)