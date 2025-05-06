import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO
import torch
from dataclasses import dataclass
import json

@dataclass
class Detection:
    """Class for storing detection results."""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height
    tile_pos: Tuple[int, int]  # (x, y) position of the tile

def create_tiles(image: np.ndarray, tile_size: int = 864, overlap: float = 0.15) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Create overlapping tiles from an image.
    
    Args:
        image: Input image
        tile_size: Size of square tiles
        overlap: Overlap ratio between tiles
        
    Returns:
        List of (tile, (x, y)) tuples where (x, y) is the top-left corner of the tile
    """
    height, width = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, (x, y)))
    
    return tiles

def convert_to_original_coords(detection: Detection, tile_size: int, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert detection coordinates from tile space to original image space.
    
    Args:
        detection: Detection object
        tile_size: Size of tile
        img_width, img_height: Original image dimensions
        
    Returns:
        Tuple of (x_center, y_center, width, height) in original image coordinates
    """
    x_center, y_center, width, height = detection.bbox
    tile_x, tile_y = detection.tile_pos
    
    # Convert to absolute coordinates in tile
    x_abs = x_center * tile_size
    y_abs = y_center * tile_size
    
    # Convert to original image coordinates
    x_orig = (x_abs + tile_x) / img_width
    y_orig = (y_abs + tile_y) / img_height
    width_orig = width * tile_size / img_width
    height_orig = height * tile_size / img_height
    
    return x_orig, y_orig, width_orig, height_orig

def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Perform non-maximum suppression on detections.
    
    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of filtered Detection objects
    """
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    # Convert to numpy arrays for easier computation
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = np.arange(len(detections))
    
    keep = []
    while indices.size > 0:
        # Get the highest confidence detection
        i = indices[0]
        keep.append(i)
        
        if indices.size == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = []
        for j in indices[1:]:
            box1 = boxes[i]
            box2 = boxes[j]
            
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
                iou = 0
            else:
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                iou = intersection / union
            
            ious.append(iou)
        
        # Remove boxes with IoU > threshold
        ious = np.array(ious)
        indices = indices[1:][ious <= iou_threshold]
    
    return [detections[i] for i in keep]

def process_image(model: YOLO,
                 image: np.ndarray,
                 tile_size: int = 864,
                 overlap: float = 0.15,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.5,
                 device: str = '0') -> List[Detection]:
    """
    Process a full-size image using tiling and NMS.
    
    Args:
        model: YOLOv8 model
        image: Input image
        tile_size: Size of square tiles
        overlap: Overlap ratio between tiles
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: Device to use for inference
        
    Returns:
        List of Detection objects after NMS
    """
    height, width = image.shape[:2]
    
    # Create tiles
    tiles = create_tiles(image, tile_size, overlap)
    
    # Process each tile
    all_detections = []
    for tile, (x, y) in tiles:
        # Run inference
        results = model(tile, conf=conf_threshold, device=device)[0]
        
        # Process detections
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xywh[0].tolist()  # x_center, y_center, width, height
            
            # Create detection object
            detection = Detection(
                class_id=class_id,
                confidence=confidence,
                bbox=bbox,
                tile_pos=(x, y)
            )
            all_detections.append(detection)
    
    # Convert coordinates to original image space
    for detection in all_detections:
        detection.bbox = convert_to_original_coords(detection, tile_size, width, height)
    
    # Apply NMS
    filtered_detections = non_max_suppression(all_detections, iou_threshold)
    
    return filtered_detections

def visualize_detections(image: np.ndarray,
                        detections: List[Detection],
                        class_names: Dict[int, str],
                        output_path: str = None) -> np.ndarray:
    """
    Visualize detections on image.
    
    Args:
        image: Input image
        detections: List of Detection objects
        class_names: Dictionary mapping class IDs to names
        output_path: Path to save visualization (optional)
        
    Returns:
        Image with visualizations
    """
    # Create copy of image
    vis_image = image.copy()
    
    # Define colors for each class
    colors = {
        0: (0, 255, 0),    # Green for Macrolophus
        1: (255, 0, 0),    # Blue for Nesidiocoris
        2: (0, 0, 255)     # Red for Whiteflies
    }
    
    # Draw each detection
    for det in detections:
        x_center, y_center, width, height = det.bbox
        x1 = int((x_center - width/2) * image.shape[1])
        y1 = int((y_center - height/2) * image.shape[0])
        x2 = int((x_center + width/2) * image.shape[1])
        y2 = int((y_center + height/2) * image.shape[0])
        
        # Draw bounding box
        color = colors.get(det.class_id, (255, 255, 255))
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[det.class_id]} {det.confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization if output path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image

def process_image_file(model_path: str,
                      image_path: str,
                      output_dir: str,
                      tile_size: int = 864,
                      overlap: float = 0.15,
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.5,
                      device: str = '0'):
    """
    Process a single image file and save results.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        output_dir: Directory to save results
        tile_size: Size of square tiles
        overlap: Overlap ratio between tiles
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: Device to use for inference
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Process image
    detections = process_image(
        model, image, tile_size, overlap,
        conf_threshold, iou_threshold, device
    )
    
    # Load class names
    with open(Path(model_path).parent / 'class_mapping.json', 'r') as f:
        class_names = {int(k): v for k, v in json.load(f).items()}
    
    # Visualize detections
    output_path = output_dir / f"{Path(image_path).stem}_detections.jpg"
    visualize_detections(image, detections, class_names, str(output_path))
    
    # Save detections to JSON
    detections_json = []
    for det in detections:
        detections_json.append({
            'class_id': det.class_id,
            'class_name': class_names[det.class_id],
            'confidence': det.confidence,
            'bbox': det.bbox
        })
    
    with open(output_dir / f"{Path(image_path).stem}_detections.json", 'w') as f:
        json.dump(detections_json, f, indent=4)
    
    return detections 