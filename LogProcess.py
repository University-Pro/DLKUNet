import re

# 定义日志文件路径
log_file = 'result/AttentionUNet/ACDC/Test/Test.log'

# 用于存储提取出的mean_dice、mean_hd95数值和对应的pth文件
mean_dice_results = []

# 打开并读取日志文件
with open(log_file, 'r') as file:
    logs = file.read()

# 使用正则表达式查找所有的模型加载路径
model_paths = re.findall(r"Model loaded from (.+?\.pth)", logs)

# 使用正则表达式匹配 "best val model: mean_dice" 和 "mean_hd95" 后面的数值
mean_dice_hd95_values = re.findall(r'best val model: mean_dice\s*:\s*([\d.]+)\s*mean_hd95\s*:\s*([\d.]+)', logs)

# 检查模型加载路径的数量是否与mean_dice结果数量匹配
if len(model_paths) != len(mean_dice_hd95_values):
    print("Warning: 模型加载路径数量与mean_dice和mean_hd95结果数量不一致，可能存在解析问题。")

# 将mean_dice和mean_hd95值与对应的pth文件存储在一起
for i, (dice, hd95) in enumerate(mean_dice_hd95_values):
    model_path = model_paths[i] if i < len(model_paths) else "Unknown Model"
    mean_dice_results.append((model_path, float(dice) * 100, float(hd95)))

# 输出提取的mean_dice、mean_hd95数值及其对应的pth文件
for model_path, mean_dice, mean_hd95 in mean_dice_results:
    print(f"{model_path} -> mean_dice: {mean_dice}% , mean_hd95: {mean_hd95}")

# 统计并输出最大的mean_dice与mean_hd95
max_mean_dice = max(mean_dice_results, key=lambda x: x[1])
max_mean_hd95 = min(mean_dice_results, key=lambda x: x[2])  # 因为hd95值越小越好，取最小值

print("\n最大 mean_dice:")
print(f"{max_mean_dice[0]} -> mean_dice: {max_mean_dice[1]}% , mean_hd95: {max_mean_dice[2]}")

print("\n最小 mean_hd95:")
print(f"{max_mean_hd95[0]} -> mean_dice: {max_mean_hd95[1]}% , mean_hd95: {max_mean_hd95[2]}")
