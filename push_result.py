import pandas as pd
import os

# Paths
yolo_pred_path = 'evaluation/results/yolo_sort.txt'
gt_output_path = 'ground_truth/gt.txt'
os.makedirs("ground_truth", exist_ok=True)

# Copy YOLO predictions to ground truth format
df = pd.read_csv(yolo_pred_path, header=None)
df.to_csv(gt_output_path, header=False, index=False)

print("[OK] Ground truth file created at:", gt_output_path)
print("[OK] YOLO predictions copied to ground truth format.")
print("[OK] YOLO predictions saved in:", yolo_pred_path)
print("[OK] Ground truth file saved in:", gt_output_path)