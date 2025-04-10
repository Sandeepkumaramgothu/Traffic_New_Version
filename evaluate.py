import motmetrics as mm
import pandas as pd
import os

def load_mot_file(path):
    return mm.io.loadtxt(open(path), fmt="mot15-2", min_confidence=0.0)

def evaluate(gt_path, pred_path, tracker_name):
    acc = mm.MOTAccumulator(auto_id=True)

    gt = load_mot_file(gt_path)
    pred = load_mot_file(pred_path)

    for frame in sorted(gt['FrameId'].unique()):
        gt_frame = gt[gt['FrameId'] == frame]
        pred_frame = pred[pred['FrameId'] == frame]

        gt_ids = gt_frame['Id'].tolist()
        gt_boxes = list(zip(gt_frame['X'], gt_frame['Y'], gt_frame['Width'], gt_frame['Height']))
        pred_ids = pred_frame['Id'].tolist()
        pred_boxes = list(zip(pred_frame['X'], pred_frame['Y'], pred_frame['Width'], pred_frame['Height']))

        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=tracker_name)
    return summary

def main():
    gt_path = 'ground_truth/gt.txt'

    results = {
        "YOLO+SORT": "evaluation/results/yolo_sort.txt",
        "V2 Tracker": "evaluation/results/v2_vehicle.txt"
    }

    accs = {}
    for name, pred_path in results.items():
        if os.path.exists(pred_path):
            accs[name] = evaluate(gt_path, pred_path, name)
        else:
            print(f"Prediction file missing: {pred_path}")

    mh = mm.metrics.create()
    summary = mm.io.render_summary(accs, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(summary)

if __name__ == "__main__":
    main()
