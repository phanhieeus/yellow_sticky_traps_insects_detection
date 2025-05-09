import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import argparse

def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Đọc file label YOLO, trả về list (class_id, x_center, y_center, w, h)
    """
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x, y, w, h = map(float, parts)
                labels.append((int(class_id), x, y, w, h))
    return labels

def iou(box1, box2):
    # box: (x_center, y_center, w, h) normalized
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # convert to xyxy
    xa1 = x1 - w1/2
    ya1 = y1 - h1/2
    xa2 = x1 + w1/2
    ya2 = y1 + h1/2
    xb1 = x2 - w2/2
    yb1 = y2 - h2/2
    xb2 = x2 + w2/2
    yb2 = y2 + h2/2
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (xa2 - xa1) * (ya2 - ya1)
    area2 = (xb2 - xb1) * (yb2 - yb1)
    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

def match_detections(gt, preds, iou_thr=0.5):
    """
    Ghép nhãn dự đoán với ground truth, trả về TP, FP, FN cho từng class
    gt: list (class_id, x, y, w, h)
    preds: list (class_id, conf, x, y, w, h)
    """
    gt_by_class = defaultdict(list)
    for g in gt:
        gt_by_class[g[0]].append(g[1:])
    pred_by_class = defaultdict(list)
    for p in preds:
        pred_by_class[p[0]].append(p[1:])
    results = {}
    for class_id in set(list(gt_by_class.keys()) + list(pred_by_class.keys())):
        gt_boxes = gt_by_class[class_id]
        pred_boxes = pred_by_class[class_id]
        matched = set()
        tp = 0
        fp = 0
        used_gt = set()
        pred_boxes = sorted(pred_boxes, key=lambda x: -x[0])  # sort by conf desc
        for pred in pred_boxes:
            best_iou = 0
            best_gt = -1
            for i, gt_box in enumerate(gt_boxes):
                if i in used_gt:
                    continue
                iou_val = iou(pred[1:], gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = i
            if best_iou >= iou_thr and best_gt != -1:
                tp += 1
                used_gt.add(best_gt)
            else:
                fp += 1
        fn = len(gt_boxes) - len(used_gt)
        results[class_id] = {'tp': tp, 'fp': fp, 'fn': fn}
    return results

def compute_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_ap(recalls, precisions):
    """Tính AP từ các điểm recall, precision (VOC 2007 11-point)"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = [prec for r, prec in zip(recalls, precisions) if r >= t]
        p = max(p) if p else 0
        ap += p / 11.0
    return ap

def evaluate_folder(gt_dir, pred_dir, class_names, iou_thr=0.5, conf_thr=0.001):
    """
    Đánh giá toàn bộ folder test: tính mAP@0.5, precision, recall, f1 cho từng lớp
    """
    all_gts = defaultdict(list)
    all_preds = defaultdict(list)
    image_ids = []
    for label_file in sorted(Path(gt_dir).glob('*.txt')):
        img_id = label_file.stem
        image_ids.append(img_id)
        gt = load_yolo_labels(str(label_file))
        all_gts[img_id] = gt
        # Đọc predict
        pred_file = Path(pred_dir) / f"{img_id}_det.json"
        preds = []
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                for d in json.load(f):
                    preds.append((d['class_id'], d['confidence'], *d['bbox']))
        # Lọc theo conf threshold
        preds = [p for p in preds if p[1] >= conf_thr]
        all_preds[img_id] = preds
    # Gom tất cả predict theo class để tính AP
    aps = {}
    prf1 = {}
    for class_id, class_name in class_names.items():
        # Tập hợp tất cả predict của class này trên toàn bộ ảnh
        class_preds = []
        class_gts = 0
        for img_id in image_ids:
            gt = [b for b in all_gts[img_id] if b[0] == class_id]
            preds = [p for p in all_preds[img_id] if p[0] == class_id]
            class_gts += len(gt)
            for p in preds:
                class_preds.append({'img_id': img_id, 'conf': p[1], 'bbox': p[2:]})
        # Sort by conf desc
        class_preds = sorted(class_preds, key=lambda x: -x['conf'])
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        used = defaultdict(set)
        for i, pred in enumerate(class_preds):
            gt_boxes = [b[1:] for b in all_gts[pred['img_id']] if b[0] == class_id]
            best_iou = 0
            best_gt = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in used[pred['img_id']]:
                    continue
                iou_val = iou(pred['bbox'], gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = j
            if best_iou >= iou_thr and best_gt != -1:
                tp[i] = 1
                used[pred['img_id']].add(best_gt)
            else:
                fp[i] = 1
        # Precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / class_gts if class_gts > 0 else np.zeros_like(tp_cum)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
        ap = compute_ap(recalls, precisions)
        aps[class_name] = ap
        # Tổng hợp TP, FP, FN để tính precision, recall, f1
        n_tp = int(tp.sum())
        n_fp = int(fp.sum())
        n_fn = class_gts - n_tp
        precision, recall, f1 = compute_precision_recall_f1(n_tp, n_fp, n_fn)
        prf1[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
    mAP = np.mean(list(aps.values()))
    return {'mAP@0.5': mAP, 'AP_per_class': aps, 'PRF1_per_class': prf1}

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection results (YOLO format)')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth label folder (YOLO .txt)')
    parser.add_argument('--pred', type=str, required=True, help='Path to detection results folder (json)')
    parser.add_argument('--output', type=str, default='eval_results.json', help='Output JSON file')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold for predictions')
    args = parser.parse_args()
    # Định nghĩa class_names
    class_names = {0: 'WF', 1: 'MR', 2: 'NC'}
    results = evaluate_folder(args.gt, args.pred, class_names, iou_thr=args.iou, conf_thr=args.conf)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main() 