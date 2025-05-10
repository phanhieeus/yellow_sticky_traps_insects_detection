import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO
import torch
from dataclasses import dataclass
import json
import argparse

@dataclass
class Detection:
    """Class for storing detection results."""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height
    tile_pos: Tuple[int, int]  # (x, y) position of the tile

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
        0: (0, 255, 0),    # Green for WF (Whiteflies)
        1: (255, 0, 0),    # Red for MR (Macrolophus)
        2: (0, 0, 255)     # Blue for NC (Nesidiocoris)
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