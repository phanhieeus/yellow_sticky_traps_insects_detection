import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

@dataclass
class GroundTruth:
    """Class for storing ground truth annotations."""
    class_id: int
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height

def load_ground_truth(label_path: str) -> List[GroundTruth]:
    """
    Load ground truth annotations from YOLO format label file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of GroundTruth objects
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, *bbox = map(float, line.strip().split())
            annotations.append(GroundTruth(int(class_id), tuple(bbox)))
    return annotations

def calculate_iou(box1: Tuple[float, float, float, float],
                 box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union between two boxes.
    
    Args:
        box1, box2: Boxes in format (x_center, y_center, width, height)
        
    Returns:
        IoU value
    """
    # Convert to (x1, y1, x2, y2) format
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union

def evaluate_detections(gt_annotations: List[GroundTruth],
                       detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
                       iou_threshold: float = 0.5) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Evaluate detections against ground truth.
    
    Args:
        gt_annotations: List of ground truth annotations
        detections: List of (class_id, confidence, bbox) tuples
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (true_positives, false_positives, false_negatives) per class
    """
    # Sort detections by confidence
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    # Initialize counters
    true_positives = {0: 0, 1: 0, 2: 0}  # Macrolophus, Nesidiocoris, Whiteflies
    false_positives = {0: 0, 1: 0, 2: 0}
    false_negatives = {0: 0, 1: 0, 2: 0}
    
    # Track matched ground truth boxes
    matched_gt = set()
    
    # Process each detection
    for det_class_id, det_conf, det_bbox in detections:
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for i, gt in enumerate(gt_annotations):
            if i in matched_gt:
                continue
                
            if gt.class_id == det_class_id:
                iou = calculate_iou(det_bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        # Update counters
        if best_iou >= iou_threshold:
            true_positives[det_class_id] += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives[det_class_id] += 1
    
    # Count unmatched ground truth boxes as false negatives
    for i, gt in enumerate(gt_annotations):
        if i not in matched_gt:
            false_negatives[gt.class_id] += 1
    
    return true_positives, false_positives, false_negatives

def calculate_metrics(true_positives: Dict[int, int],
                     false_positives: Dict[int, int],
                     false_negatives: Dict[int, int]) -> Dict[int, Dict[str, float]]:
    """
    Calculate precision, recall, and F1-score for each class.
    
    Args:
        true_positives, false_positives, false_negatives: Counts per class
        
    Returns:
        Dictionary of metrics per class
    """
    metrics = {}
    for class_id in true_positives.keys():
        tp = true_positives[class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return metrics

def plot_confusion_matrix(y_true: List[int],
                         y_pred: List[int],
                         class_names: Dict[int, str],
                         output_path: str):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: List of true class labels
        y_pred: List of predicted class labels
        class_names: Dictionary mapping class IDs to names
        output_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] for i in range(len(class_names))],
                yticklabels=[class_names[i] for i in range(len(class_names))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model_path: str,
                  test_dir: str,
                  output_dir: str,
                  class_names: Dict[int, str],
                  iou_threshold: float = 0.5,
                  conf_threshold: float = 0.25,
                  device: str = '0'):
    """
    Evaluate model on test set and generate metrics and visualizations.
    
    Args:
        model_path: Path to trained model weights
        test_dir: Directory containing test images and labels
        output_dir: Directory to save evaluation results
        class_names: Dictionary mapping class IDs to names
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for detections
        device: Device to use for inference
    """
    from src.inference import process_image_file
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize counters
    true_positives = {0: 0, 1: 0, 2: 0}
    false_positives = {0: 0, 1: 0, 2: 0}
    false_negatives = {0: 0, 1: 0, 2: 0}
    
    # Lists for confusion matrix
    y_true = []
    y_pred = []
    
    # Process each test image
    test_dir = Path(test_dir)
    for img_path in tqdm(list((test_dir / 'images').glob('*.jpg'))):
        # Load ground truth
        label_path = test_dir / 'labels' / img_path.with_suffix('.txt').name
        gt_annotations = load_ground_truth(str(label_path))
        
        # Get predictions
        detections = process_image_file(
            model_path, str(img_path), str(output_dir / 'detections'),
            conf_threshold=conf_threshold, device=device
        )
        
        # Convert detections to format for evaluation
        detections_eval = [(d.class_id, d.confidence, d.bbox) for d in detections]
        
        # Evaluate detections
        tp, fp, fn = evaluate_detections(gt_annotations, detections_eval, iou_threshold)
        
        # Update counters
        for class_id in tp.keys():
            true_positives[class_id] += tp[class_id]
            false_positives[class_id] += fp[class_id]
            false_negatives[class_id] += fn[class_id]
        
        # Update confusion matrix data
        for gt in gt_annotations:
            y_true.append(gt.class_id)
            # Find best matching detection
            best_iou = 0
            best_class = -1
            for det in detections:
                if det.class_id == gt.class_id:
                    iou = calculate_iou(gt.bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_class = det.class_id
            y_pred.append(best_class if best_iou >= iou_threshold else -1)
    
    # Calculate metrics
    metrics = calculate_metrics(true_positives, false_positives, false_negatives)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, str(output_dir / 'confusion_matrix.png'))
    
    return metrics 