"""
find best mean_dice and mean_hd95 from log file
"""
import re
log_file = 'result/AttentionUNet/ACDC/Test/Test.log'
mean_dice_results = []

with open(log_file, 'r') as file:
    logs = file.read()

model_paths = re.findall(r"Model loaded from (.+?\.pth)", logs)
mean_dice_hd95_values = re.findall(r'best val model: mean_dice\s*:\s*([\d.]+)\s*mean_hd95\s*:\s*([\d.]+)', logs)

if len(model_paths) != len(mean_dice_hd95_values):
    print("Warning: the number of model paths and mean_dice/mean_hd95 values do not match.")

for i, (dice, hd95) in enumerate(mean_dice_hd95_values):
    model_path = model_paths[i] if i < len(model_paths) else "Unknown Model"
    mean_dice_results.append((model_path, float(dice) * 100, float(hd95)))

for model_path, mean_dice, mean_hd95 in mean_dice_results:
    print(f"{model_path} -> mean_dice: {mean_dice}% , mean_hd95: {mean_hd95}")

# 统计并输出最大的mean_dice与mean_hd95
max_mean_dice = max(mean_dice_results, key=lambda x: x[1])
max_mean_hd95 = min(mean_dice_results, key=lambda x: x[2])

print("\nmax mean_dice:")
print(f"{max_mean_dice[0]} -> mean_dice: {max_mean_dice[1]}% , mean_hd95: {max_mean_dice[2]}")

print("\nminimum mean_hd95:")
print(f"{max_mean_hd95[0]} -> mean_dice: {max_mean_hd95[1]}% , mean_hd95: {max_mean_hd95[2]}")
