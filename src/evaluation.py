import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import argparse
import cv2
from tqdm import tqdm

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

def load_detection_json(json_path: str) -> List[Dict]:
    """
    Đọc file JSON kết quả detection
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data.get('detections', [])

def convert_yolo_to_xyxy(box, img_width, img_height):
    """
    Chuyển đổi từ (x_center, y_center, w, h) normalized sang (x1, y1, x2, y2) absolute
    """
    x_center, y_center, w, h = box
    x1 = (x_center - w/2) * img_width
    y1 = (y_center - h/2) * img_height
    x2 = (x_center + w/2) * img_width
    y2 = (y_center + h/2) * img_height
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """
    Tính IoU giữa 2 box dạng (x1, y1, x2, y2)
    """
    # Tính tọa độ giao nhau
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Tính diện tích giao nhau
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Tính diện tích của mỗi box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Tính diện tích hợp
    union = box1_area + box2_area - intersection

    # Tính IoU
    iou = intersection / union if union > 0 else 0
    return iou

def evaluate_detections(gt_labels: List[Tuple], detections: List[Dict], image_path: str, iou_threshold: float = 0.5) -> Dict:
    """
    Đánh giá kết quả detection với ground truth
    """
    # Đọc kích thước ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_height, img_width = img.shape[:2]
    
    # Chuyển đổi ground truth sang định dạng (class_id, [x1, y1, x2, y2])
    gt_boxes = [(label[0], convert_yolo_to_xyxy(label[1:], img_width, img_height)) for label in gt_labels]
    
    # Chuyển đổi detections sang định dạng (class_id, confidence, [x1, y1, x2, y2])
    det_boxes = [(det['class'], det['confidence'], det['bbox']) for det in detections]
    
    # Sắp xếp detections theo confidence giảm dần
    det_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Khởi tạo các biến đếm
    metrics = {
        'total_gt': len(gt_boxes),
        'total_detections': len(det_boxes),
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_class': defaultdict(lambda: {
            'gt_count': 0,
            'detection_count': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        })
    }
    
    # Đếm số lượng ground truth cho mỗi class
    for gt_class, _ in gt_boxes:
        metrics['per_class'][gt_class]['gt_count'] += 1
    
    # Đếm số lượng detections cho mỗi class
    for det_class, _, _ in det_boxes:
        metrics['per_class'][det_class]['detection_count'] += 1
    
    # Đánh dấu các ground truth đã được match
    matched_gt = set()
    
    # Match detections với ground truth
    for det_class, det_conf, det_box in det_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        # Tìm ground truth có IoU cao nhất
        for gt_idx, (gt_class, gt_box) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if gt_class != det_class:
                continue
                
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Nếu IoU > threshold, đây là true positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            metrics['true_positives'] += 1
            metrics['per_class'][det_class]['true_positives'] += 1
        else:
            metrics['false_positives'] += 1
            metrics['per_class'][det_class]['false_positives'] += 1
    
    # Tính false negatives
    metrics['false_negatives'] = len(gt_boxes) - len(matched_gt)
    for gt_class, _ in gt_boxes:
        if gt_class not in matched_gt:
            metrics['per_class'][gt_class]['false_negatives'] += 1
    
    # Tính các metrics
    metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Tính metrics cho từng class
    for class_id in metrics['per_class']:
        class_metrics = metrics['per_class'][class_id]
        tp = class_metrics['true_positives']
        fp = class_metrics['false_positives']
        fn = class_metrics['false_negatives']
        
        class_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_metrics['f1'] = 2 * class_metrics['precision'] * class_metrics['recall'] / (class_metrics['precision'] + class_metrics['recall']) if (class_metrics['precision'] + class_metrics['recall']) > 0 else 0
    
    return metrics

def evaluate_directory(gt_dir: str, pred_dir: str, image_dir: str, iou_threshold: float = 0.5) -> Dict:
    """
    Đánh giá kết quả detection cho toàn bộ thư mục
    """
    # Khởi tạo metrics tổng hợp
    total_metrics = {
        'total_gt': 0,
        'total_detections': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_class': defaultdict(lambda: {
            'gt_count': 0,
            'detection_count': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }),
        'per_image': {}
    }

    # Lấy danh sách các file
    gt_files = sorted(Path(gt_dir).glob('*.txt'))
    if not gt_files:
        raise ValueError(f"No label files found in {gt_dir}")

    print(f"Evaluating {len(gt_files)} images...")
    for gt_file in tqdm(gt_files):
        # Lấy tên file không có phần mở rộng
        image_id = gt_file.stem
        
        # Đường dẫn đến ảnh và kết quả detection
        image_path = Path(image_dir) / f"{image_id}.jpg"
        pred_path = Path(pred_dir) / f"{image_id}_det.json"
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
            
        if not pred_path.exists():
            print(f"Warning: Detection result not found: {pred_path}")
            continue

        # Đọc ground truth và predictions
        gt_labels = load_yolo_labels(str(gt_file))
        detections = load_detection_json(str(pred_path))
        
        # Đánh giá cho ảnh này
        metrics = evaluate_detections(gt_labels, detections, str(image_path), iou_threshold)
        
        # Cập nhật metrics tổng hợp
        total_metrics['total_gt'] += metrics['total_gt']
        total_metrics['total_detections'] += metrics['total_detections']
        total_metrics['true_positives'] += metrics['true_positives']
        total_metrics['false_positives'] += metrics['false_positives']
        total_metrics['false_negatives'] += metrics['false_negatives']
        
        # Cập nhật metrics theo class
        for class_id, class_metrics in metrics['per_class'].items():
            for key in ['gt_count', 'detection_count', 'true_positives', 'false_positives', 'false_negatives']:
                total_metrics['per_class'][class_id][key] += class_metrics[key]
        
        # Lưu metrics của từng ảnh
        total_metrics['per_image'][image_id] = metrics

    # Tính các metrics tổng hợp
    total_metrics['precision'] = total_metrics['true_positives'] / (total_metrics['true_positives'] + total_metrics['false_positives']) if (total_metrics['true_positives'] + total_metrics['false_positives']) > 0 else 0
    total_metrics['recall'] = total_metrics['true_positives'] / (total_metrics['true_positives'] + total_metrics['false_negatives']) if (total_metrics['true_positives'] + total_metrics['false_negatives']) > 0 else 0
    total_metrics['f1'] = 2 * total_metrics['precision'] * total_metrics['recall'] / (total_metrics['precision'] + total_metrics['recall']) if (total_metrics['precision'] + total_metrics['recall']) > 0 else 0

    # Tính metrics cho từng class
    for class_id in total_metrics['per_class']:
        class_metrics = total_metrics['per_class'][class_id]
        tp = class_metrics['true_positives']
        fp = class_metrics['false_positives']
        fn = class_metrics['false_negatives']
        
        class_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_metrics['f1'] = 2 * class_metrics['precision'] * class_metrics['recall'] / (class_metrics['precision'] + class_metrics['recall']) if (class_metrics['precision'] + class_metrics['recall']) > 0 else 0

    return total_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection results')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth labels directory')
    parser.add_argument('--pred', type=str, required=True, help='Path to detection results directory')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--output', type=str, default='eval_results.json', help='Output JSON file')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    args = parser.parse_args()
    
    # Đánh giá toàn bộ thư mục
    metrics = evaluate_directory(args.gt, args.pred, args.images, args.iou)
    
    # Lưu kết quả
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # In kết quả tổng hợp
    print("\nOverall Metrics:")
    print(f"Total Ground Truth Objects: {metrics['total_gt']}")
    print(f"Total Detected Objects: {metrics['total_detections']}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    class_names = {0: 'WF', 1: 'MR', 2: 'NC'}
    for class_id, class_metrics in metrics['per_class'].items():
        print(f"\nClass {class_names.get(class_id, class_id)}:")
        print(f"Ground Truth Count: {class_metrics['gt_count']}")
        print(f"Detection Count: {class_metrics['detection_count']}")
        print(f"True Positives: {class_metrics['true_positives']}")
        print(f"False Positives: {class_metrics['false_positives']}")
        print(f"False Negatives: {class_metrics['false_negatives']}")
        print(f"Precision: {class_metrics['precision']:.4f}")
        print(f"Recall: {class_metrics['recall']:.4f}")
        print(f"F1 Score: {class_metrics['f1']:.4f}")

if __name__ == '__main__':
    main() 