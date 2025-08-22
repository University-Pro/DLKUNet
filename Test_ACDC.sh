#!/bin/bash

# 该文件不能同时运行多个

# Directory containing .pth files
PTH_DIR="./result/MissFormer/ACDC/Pth"
# Directory to save logs
LOG_DIR="./result/MissFormer/ACDC/Test"
# Set patch size
PATCH_SIZE=224

# Check if LOG_DIR exists, if not create it
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Iterate over each .pth file in the directory
for MODEL_PATH in $PTH_DIR/*.pth; do
  # Extract epoch number from filename
  EPOCH=$(echo $MODEL_PATH | grep -oP '(?<=epoch_)\d+(?=_checkpoint.pth)')
  # Set log filename to include epoch number
  LOG_FILE="$LOG_DIR/Test.log"

  echo "Testing $MODEL_PATH, logging to $LOG_FILE"

  # Call the Python script with the current .pth file
  python Test_ACDC.py --model_load $MODEL_PATH --log_path $LOG_FILE --patch_size $PATCH_SIZE
done
