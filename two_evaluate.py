import pandas as pd
import numpy as np
import os

def compute_iou(boxA, boxB):
    xa1, ya1, wa, ha = boxA
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = boxB
    xb2, yb2 = xb1 + wb, yb1 + hb

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = wa * ha
    area_b = wb * hb
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_tracker(gt_path, pred_path, output_csv):
    # Load ground truth and predictions
    col_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x1', 'y1', 'x2']
    gt = pd.read_csv(gt_path, header=None, names=col_names)
    pred = pd.read_csv(pred_path, header=None, names=col_names)

    all_tp, all_fp, all_fn = 0, 0, 0
    rows = []

    # Evaluate frame by frame
    for frame_id in sorted(gt['frame'].unique()):
        gt_f = gt[gt['frame'] == frame_id]
        pred_f = pred[pred['frame'] == frame_id]

        matched_gt = set()
        tp = fp = 0

        for _, prow in pred_f.iterrows():
            pbox = (prow['x'], prow['y'], prow['w'], prow['h'])
            best_iou, best_gt_idx = 0, None
            for gi, grow in gt_f.iterrows():
                if gi in matched_gt:
                    continue
                gbox = (grow['x'], grow['y'], grow['w'], grow['h'])
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_iou >= 0.5:
                matched_gt.add(best_gt_idx)
                tp += 1
            else:
                fp += 1

        fn = len(gt_f) - len(matched_gt)
        rows.append([frame_id, tp, fp, fn])
        all_tp += tp
        all_fp += fp
        all_fn += fn

    # Save per-frame metrics
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(rows, columns=['frame', 'TP', 'FP', 'FN']).to_csv(output_csv, index=False)

    # Final summary metrics
    prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\nEvaluation for {os.path.basename(pred_path)}")
    print(f"Total True Positives : {all_tp}")
    print(f"Total False Positives: {all_fp}")
    print(f"Total False Negatives: {all_fn}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

# === Run evaluation ===
if __name__ == "__main__":
    gt_path = 'ground_truth/gt.txt'
    yolo_pred = 'evaluation/results/yolo_sort.txt'
    v2_pred = 'evaluation/results/v2_vehicle.txt'
    yolo_csv = 'evaluation/results/eval_yolo.csv'
    v2_csv = 'evaluation/results/eval_v2.csv'

    evaluate_tracker(gt_path, yolo_pred, yolo_csv)
    evaluate_tracker(gt_path, v2_pred, v2_csv)
