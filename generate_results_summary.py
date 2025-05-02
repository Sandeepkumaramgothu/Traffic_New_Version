import os
import pandas as pd
import motmetrics as mm

# === File Paths ===
gt_file = "ground_truth/gt.txt"
yolo_file = "evaluation/results/yolo_sort.txt"
v2_file = "evaluation/results/v2_vehicle.txt"

def load_mot_file(path):
    # DO NOT use 'fmt="motchallenge"' — it's unsupported in your version
    return mm.io.loadtxt(open(path), min_confidence=0.0)

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

# === Counting Accuracy ===
def capture_count_accuracy():
    print("\n=== Counting Accuracy ===")
    results = []
    for tracker_name, pred_path in [("YOLO+SORT", yolo_file), ("V2 Tracker", v2_file)]:
        if not os.path.exists(pred_path) or os.stat(pred_path).st_size == 0:
            print(f"[WARNING] Skipping {tracker_name} – file missing or empty: {pred_path}")
            continue
        try:
            gt_df = pd.read_csv(gt_file, header=None)
            pred_df = pd.read_csv(pred_path, header=None)
            gt_ids = set(gt_df[1].unique())
            pred_ids = set(pred_df[1].unique())
            gt_count = len(gt_ids)
            pred_count = len(pred_ids)
            error = abs(pred_count - gt_count)
            accuracy = 100 * (1 - error / gt_count) if gt_count > 0 else 0
            direction = "Overcounted" if pred_count > gt_count else "Undercounted" if pred_count < gt_count else "Exact"
            results.append([tracker_name, gt_count, pred_count, error, f"{accuracy:.2f}%", direction])
        except Exception as e:
            print(f"[ERROR] Failed processing {tracker_name}: {e}")
    return results

# === MOT Evaluation ===
def capture_mot_metrics():
    accs = {}
    for tracker_name, pred_path in [("YOLO+SORT", yolo_file), ("V2 Tracker", v2_file)]:
        if not os.path.exists(pred_path) or os.stat(pred_path).st_size == 0:
            print(f"[WARNING] Skipping MOT metrics for {tracker_name} – file missing or empty.")
            continue
        try:
            accs[tracker_name] = evaluate(gt_file, pred_path, tracker_name)
        except Exception as e:
            print(f"[ERROR] Failed MOT evaluation for {tracker_name}: {e}")
    if not accs:
        return None
    mh = mm.metrics.create()
    return mh.compute_many(accs.values(), names=accs.keys(), metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

# === Output Results ===
def save_results(counting_results, mot_summary, out_path="results_summary.txt"):
    with open(out_path, "w") as f:
        f.write("=== Vehicle Counting Accuracy ===\n\n")
        f.write(f"{'Tracker':<15}{'GT Count':<12}{'Predicted':<12}{'Error':<8}{'Accuracy':<12}{'Result':<15}\n")
        for row in counting_results:
            f.write(f"{row[0]:<15}{row[1]:<12}{row[2]:<12}{row[3]:<8}{row[4]:<12}{row[5]:<15}\n")
        f.write("\n")
        if mot_summary is not None:
            f.write("=== MOT Evaluation Metrics ===\n\n")
            f.write(mot_summary.to_string())
        else:
            f.write("[No MOT summary generated – possibly all input files were empty or missing.]\n")
    print(f"\n✅ Results saved to {out_path}")

# === Main ===
if __name__ == "__main__":
    counting_results = capture_count_accuracy()
    mot_summary = capture_mot_metrics()
    save_results(counting_results, mot_summary)
