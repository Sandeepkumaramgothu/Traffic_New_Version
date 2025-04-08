# If running in Jupyter/Colab:
# !pip install -qqq motmetrics pandas

import os
import pandas as pd
import motmetrics as mm

# Print the installed version of motmetrics
print("motmetrics version:", mm.__version__)


def load_mot_file(path):
    """
    Loads a MOT-formatted result or ground truth file.
    Format: frame, id, x, y, w, h, conf, -1, -1, -1
    """
    with open(path) as f:
        return mm.io.loadtxt(f, fmt="mot15-2", min_confidence=0.0)


def evaluate_tracker(gt_path, pred_path, tracker_name):
    acc = mm.MOTAccumulator(auto_id=True)

    gt = load_mot_file(gt_path)
    pred = load_mot_file(pred_path)

    for frame_id in sorted(gt['FrameId'].unique()):
        gt_frame = gt[gt['FrameId'] == frame_id]
        pred_frame = pred[pred['FrameId'] == frame_id]

        gt_ids = gt_frame['Id'].tolist()
        gt_boxes = list(zip(gt_frame['X'], gt_frame['Y'], gt_frame['Width'], gt_frame['Height']))

        pred_ids = pred_frame['Id'].tolist()
        pred_boxes = list(zip(pred_frame['X'], pred_frame['Y'], pred_frame['Width'], pred_frame['Height']))

        # Compute IoU distances (1 - IoU)
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dist)

    # Compute all standard MOT metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=tracker_name)
    return summary


def main():
    gt_path = 'ground_truth/gt.txt'  # Update this if your GT file is in a different location

    result_files = {
        "YOLOv8 + SORT": 'evaluation/results/yolo_sort.txt',
        "V2 BackgroundSub": 'evaluation/results/v2_vehicle.txt'
    }

    summaries = []

    for name, path in result_files.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        print(f" Evaluating: {name}")
        summary = evaluate_tracker(gt_path, path, name)
        summaries.append(summary)

    accs = {name: s for name, s in zip(result_files.keys(), summaries)}
    mh = mm.metrics.create()
    
    # Print nicely formatted metrics
    print("\nEvaluation Summary:")
    print(mm.io.render_summary(
        accs,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))


if __name__ == "__main__":
    main()
