"""
ACDC数据集的测试代码
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

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

# 导入数据集
from DataLoader_ACDC import ACDCdataset
from DataLoader_ACDC import RandomGenerator

# 导入网络
# from network.LKA_Source import UNet
# from network.DDL_Rewrite import Network
# from network.DDL_Source import UNet
# from network.DDL_Source_2Layer import UNet
# from network.DDL_Source_3Layer import UNet
# from network.DDL_Source_64channel import UNet
# from network.DDL_Source_128channel import UNet
# from network.Unet2D import UNet
# from network.ULite_ACDC import ULite
from network.MissFormer.MISSFormer import MISSFormer

def set_seed(seed_value=42):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_value)

def setup_logging(log_file):
    """日志记录"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])

# 查找最新的文件
def latest_checkpoint(path):
    """查找path中最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

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

def DiceLoss_test():
    # 创建模拟数据
    n_classes = 9
    dice_loss = DiceLoss(n_classes)

    # 模拟的输入和目标
    # 假设 batch_size = 1, height = width = 2, n_classes = 2
    inputs = torch.randn(1, n_classes, 2, 2)  # 模拟网络输出
    target = torch.randint(0, n_classes, (1, 2, 2))  # 模拟目标标签

    # 计算 Dice Loss
    loss = dice_loss(inputs, target)
    print(loss)
    print(loss.item())

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

def calculate_metric_percase_test():
    # 创建模拟输入
    pred = np.array([[1, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    gt = np.array([[1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

    # 计算模拟输出
    print(calculate_metric_percase(pred, gt))

# 下面是修改代码，能够一定程度上提高测试计算速度
def test_single_volume(image, label, net, classes, patch_size=[512, 512], test_save_path=None, case=None, z_spacing=1):
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 读取图像内容并传输到GPU
    image, label = image.squeeze(0).to(device), label.squeeze(0).to(device)
    
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

    # 设置test数据集
    db_test = ACDCdataset(base_dir="./datasets/ACDC", list_dir="./datasets/ACDC/lists_ACDC", split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4) # 这里设置batch_size = 1,num_workers = 4
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # 注意和Synapse的不同

        metric_i = test_single_volume(image, label, model, classes=4, patch_size=[patch_size, patch_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(db_test)

    # range从1-4
    for i in range(1, 4):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info('Testing Finished')

if __name__=='__main__':
    # 参数设计
    parser = argparse.ArgumentParser(description='Test a deep learning model on a given dataset and savecheckpoint')
    parser.add_argument("--model_load",type=str,default='./models_log/MambaV0/Pth/model_epoch_150_checkpoint.pth',help='The path of the checkpoint')
    parser.add_argument("--log_path",type=str,default='./models_log/MambaV0/Test/running.log',help='The path of the log')
    parser.add_argument("--patch_size",type=int,default=224,help='The resolution of test image size')
    parser.add_argument("--test_save_path",type=str,default=None,help="The path to save segmentation results")
    
    option = parser.parse_args()
    
    # 设定日志位置
    setup_logging(option.log_path)
    logging.info(f"log file will be setted at {option.log_path}")

    # 检查GPU是否可用
    # 检查是否有可用的GPU，如果有，则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Now is Going to use {device.type}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")


    # 创建模型实例
    # model = ULite().to(device=device)
    # model = UNet(n_channels=1, n_classes=4).to(device)
    # model = Network(in_channel=1,out_channel=96,final_channel=4).to(device=device)
    model = MISSFormer(num_classes=4).to(device=device)

    # 加载权重
    # model.load_state_dict(torch.load(option.model_load, map_location=device))
    model = load_model(model, option.model_load, device) # 新版本,自动处理单卡和多卡模型


    logging.info(f"Model loaded from {option.model_load}")

    # 调用测试函数
    inference(model, test_save_path=option.test_save_path,log_path=option.log_path,patch_size=option.patch_size)