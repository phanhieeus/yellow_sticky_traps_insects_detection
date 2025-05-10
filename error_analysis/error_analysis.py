import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format labels"""
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
    """Load detection results from JSON file"""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data.get('detections', [])

def convert_yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO format to XYXY format"""
    x_center, y_center, w, h = box
    x1 = (x_center - w/2) * img_width
    y1 = (y_center - h/2) * img_height
    x2 = (x_center + w/2) * img_width
    y2 = (y_center + h/2) * img_height
    return [x1, y1, x2, y2]

def calculate_image_metrics(gt_labels: List[Tuple], detections: List[Dict], img_width: int, img_height: int) -> Dict:
    """Calculate metrics for a single image"""
    # Convert ground truth to XYXY format
    gt_boxes = [(label[0], convert_yolo_to_xyxy(label[1:], img_width, img_height)) for label in gt_labels]
    
    # Convert detections to XYXY format
    det_boxes = [(det['class'], det['confidence'], det['bbox']) for det in detections]
    
    # Initialize metrics
    metrics = {
        'total_gt': len(gt_boxes),
        'total_detections': len(det_boxes),
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_class': {}
    }
    
    # Match detections with ground truth using IoU
    matched_gt = set()
    for det_class, det_conf, det_box in det_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_class, gt_box) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if gt_class != det_class:
                continue
            
            # Calculate IoU
            x1 = max(det_box[0], gt_box[0])
            y1 = max(det_box[1], gt_box[1])
            x2 = min(det_box[2], gt_box[2])
            y2 = min(det_box[3], gt_box[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            union = det_area + gt_area - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= 0.5 and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            metrics['true_positives'] += 1
        else:
            metrics['false_positives'] += 1
    
    metrics['false_negatives'] = len(gt_boxes) - len(matched_gt)
    
    # Calculate precision, recall, and F1
    metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    return metrics

def visualize_image(image_path: str, gt_labels: List[Tuple], detections: List[Dict], metrics: Dict, output_path: str):
    """Visualize image with ground truth and predictions"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Class names mapping
    class_names = {0: 'WF', 1: 'MR', 2: 'NC'}
    
    # Convert ground truth to XYXY format
    gt_boxes = [(label[0], convert_yolo_to_xyxy(label[1:], img_width, img_height)) for label in gt_labels]
    
    # Draw ground truth boxes (green)
    for class_id, box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add background for text
        text = f"GT_{class_names[class_id]}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Draw detection boxes (red)
    for det in detections:
        box = det['bbox']
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Add background for text
        text = f"Pred_{class_names[det['class']]} ({det['confidence']:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), (0, 0, 255), -1)
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add metrics text with background
    metrics_text = f"F1: {metrics['f1']:.3f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}"
    (text_width, text_height), _ = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    # Add semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10+text_width+20, 10+text_height+20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    # Add text
    cv2.putText(img, metrics_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Save visualization
    cv2.imwrite(output_path, img)

def analyze_worst_performing_images(n: int = 5, metric: str = 'f1'):
    """
    Analyze and visualize n worst performing images based on specified metric
    Args:
        n: Number of worst performing images to analyze
        metric: Metric to sort by ('f1', 'precision', 'recall')
    """
    # Load evaluation results
    with open('inference_results/eval_results.json', 'r') as f:
        eval_results = json.load(f)
    
    # Get per-image metrics
    image_metrics = []
    for image_id, metrics in eval_results['per_image'].items():
        metrics['image_id'] = image_id
        image_metrics.append(metrics)
    
    # Sort by specified metric
    image_metrics.sort(key=lambda x: x[metric])
    worst_images = image_metrics[:n]
    
    # Create output directory
    output_dir = Path('error_analysis/worst_performing')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize worst performing images
    test_dir = Path('data/data_for_train_test/test')
    for metrics in worst_images:
        image_id = metrics['image_id']
        gt_path = test_dir / 'labels' / f"{image_id}.txt"
        pred_path = Path('inference_results') / f"{image_id}_det.json"
        image_path = test_dir / 'images' / f"{image_id}.jpg"
        output_path = output_dir / f"{image_id}_analysis.jpg"
        
        gt_labels = load_yolo_labels(str(gt_path))
        detections = load_detection_json(str(pred_path))
        
        visualize_image(str(image_path), gt_labels, detections, metrics, str(output_path))
        
        print(f"\nImage {image_id} metrics:")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze worst performing images based on specified metric')
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'precision', 'recall'],
                      help='Metric to sort by (f1, precision, recall)')
    parser.add_argument('--n', type=int, default=5,
                      help='Number of worst performing images to analyze')
    
    args = parser.parse_args()
    
    print(f"\nAnalyzing {args.n} worst performing images based on {args.metric.upper()} score...")
    analyze_worst_performing_images(n=args.n, metric=args.metric) 