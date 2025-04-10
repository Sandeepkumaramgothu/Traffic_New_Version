import pandas as pd

def count_accuracy(gt_path, pred_path, tracker_name):
    gt_df = pd.read_csv(gt_path, header=None)
    pred_df = pd.read_csv(pred_path, header=None)

    gt_ids = set(gt_df[1].unique())
    pred_ids = set(pred_df[1].unique())

    gt_count = len(gt_ids)
    pred_count = len(pred_ids)
    error = abs(pred_count - gt_count)

    accuracy = 100 * (1 - error / gt_count) if gt_count > 0 else 0
    direction = "Overcounted" if pred_count > gt_count else "Undercounted" if pred_count < gt_count else "Exact"

    print(f"\n=== {tracker_name} Counting Accuracy ===")
    print(f"Ground Truth Vehicle Count : {gt_count}")
    print(f"Predicted Vehicle Count    : {pred_count}")
    print(f"Count Error                : {error}")
    print(f"Count Accuracy (%)         : {accuracy:.2f}")
    print(f"Result                     : {direction}")

if __name__ == "__main__":
    gt_file = "ground_truth/gt.txt"
    yolo_file = "evaluation/results/yolo_sort.txt"
    v2_file = "evaluation/results/v2_vehicle.txt"

    count_accuracy(gt_file, yolo_file, "YOLO+SORT")
    count_accuracy(gt_file, v2_file, "V2 Tracker")

