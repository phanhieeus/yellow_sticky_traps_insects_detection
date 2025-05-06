import os
import cv2
import numpy as np
import xmltodict
from pathlib import Path
from typing import Dict, List, Tuple
import json

def rotate_image(image: np.ndarray) -> np.ndarray:
    """
    Rotate image 90 degrees clockwise.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Rotated image
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def parse_voc_annotation(xml_path: str) -> Dict:
    """
    Parse VOC format XML annotation file.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        Dictionary containing annotation data
    """
    with open(xml_path, 'r') as f:
        xml_dict = xmltodict.parse(f.read())
    
    return xml_dict

def convert_bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, 
                        img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert VOC format bbox (x1,y1,x2,y2) to YOLO format (x_center, y_center, width, height).
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        img_width, img_height: Image dimensions
        
    Returns:
        Tuple of (x_center, y_center, width, height) in normalized coordinates
    """
    # Convert to YOLO format
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height

def process_annotations(xml_path: str, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """
    Process VOC annotations and convert to YOLO format.
    
    Args:
        xml_path: Path to XML annotation file
        img_width, img_height: Image dimensions
        
    Returns:
        List of tuples (class_id, x_center, y_center, width, height)
    """
    xml_dict = parse_voc_annotation(xml_path)
    
    # Class mapping
    class_mapping = {
        'Macrolophus': 0,
        'Nesidiocoris': 1,
        'Whiteflies': 2
    }
    
    annotations = []
    
    # Handle both single and multiple objects
    objects = xml_dict['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]
    
    for obj in objects:
        class_name = obj['name']
        if class_name not in class_mapping:
            continue
            
        class_id = class_mapping[class_name]
        bbox = obj['bndbox']
        
        # Get coordinates
        x1 = float(bbox['xmin'])
        y1 = float(bbox['ymin'])
        x2 = float(bbox['xmax'])
        y2 = float(bbox['ymax'])
        
        # Convert to YOLO format
        x_center, y_center, width, height = convert_bbox_to_yolo(x1, y1, x2, y2, img_width, img_height)
        
        annotations.append((class_id, x_center, y_center, width, height))
    
    return annotations

def process_dataset(data_dir: str, output_dir: str):
    """
    Process the entire dataset: rotate images and convert annotations.
    
    Args:
        data_dir: Directory containing original dataset
        output_dir: Directory to save processed data
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories for processed data
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'labels').mkdir(exist_ok=True)
    
    # Process each image and its annotation
    for img_path in data_dir.glob('*.jpg'):
        # Read and rotate image
        img = cv2.imread(str(img_path))
        img = rotate_image(img)
        
        # Get new dimensions after rotation
        height, width = img.shape[:2]
        
        # Save rotated image
        output_img_path = output_dir / 'images' / img_path.name
        cv2.imwrite(str(output_img_path), img)
        
        # Process corresponding annotation
        xml_path = img_path.with_suffix('.xml')
        if xml_path.exists():
            annotations = process_annotations(str(xml_path), width, height)
            
            # Save YOLO format annotations
            output_label_path = output_dir / 'labels' / img_path.with_suffix('.txt').name
            with open(output_label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    # Save class mapping
    class_mapping = {
        'Macrolophus': 0,
        'Nesidiocoris': 1,
        'Whiteflies': 2
    }
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4) 