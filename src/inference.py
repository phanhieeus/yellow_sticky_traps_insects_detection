import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from ultralytics import YOLO
import torch
from dataclasses import dataclass
import json
import argparse
import logging
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Class for storing detection results."""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height
    tile_pos: Tuple[int, int]  # (x, y) position of the tile

class InsectDetector:
    """Class for handling insect detection using YOLO model."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, patch_size: int = 864):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections
            patch_size: Size of image patches for processing
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.patch_size = patch_size
        self.class_names = {
            0: "WF",  # Whiteflies
            1: "MR",  # Macrolophus
            2: "NC"   # Nesidiocoris
        }
    
    def create_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split image into patches of size patch_size x patch_size.
        
        Args:
            image: Input image
            
        Returns:
            List of patches and their positions
        """
        height, width = image.shape[:2]
        patches = []
        positions = []
        
        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                # Calculate patch dimensions
                patch_height = min(self.patch_size, height - y)
                patch_width = min(self.patch_size, width - x)
                
                # Extract patch
                patch = image[y:y+patch_height, x:x+patch_width]
                patches.append(patch)
                positions.append((x, y))
                
        return patches, positions
    
    def merge_patches(self, image: np.ndarray, patches: List[np.ndarray], positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Merge processed patches back into original image.
        
        Args:
            image: Original image
            patches: List of processed patches
            positions: List of patch positions
            
        Returns:
            Merged image
        """
        merged = image.copy()
        for patch, (x, y) in zip(patches, positions):
            h, w = patch.shape[:2]
            merged[y:y+h, x:x+w] = patch
        return merged
    
    def convert_to_global_coords(self, detections: List[Detection], tile_pos: Tuple[int, int], 
                               patch_shape: Tuple[int, int], image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Convert patch-relative coordinates to global image coordinates.
        
        Args:
            detections: List of detections in patch coordinates
            tile_pos: Position of the patch in the original image
            patch_shape: Shape of the patch
            image_shape: Shape of the original image
            
        Returns:
            List of detections in global coordinates
        """
        global_detections = []
        tile_x, tile_y = tile_pos
        patch_h, patch_w = patch_shape
        img_h, img_w = image_shape
        
        for det in detections:
            x_center, y_center, width, height = det.bbox
            
            # Convert to absolute coordinates in patch
            abs_x = x_center * patch_w
            abs_y = y_center * patch_h
            
            # Convert to global coordinates
            global_x = (abs_x + tile_x) / img_w
            global_y = (abs_y + tile_y) / img_h
            
            global_detections.append(Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=(global_x, global_y, width, height),
                tile_pos=tile_pos
            ))
            
        return global_detections
    
    def non_max_suppression(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Calculate IoU with remaining detections
            ious = []
            for det in detections:
                if det.class_id != current.class_id:
                    ious.append(0)
                    continue
                    
                # Convert to absolute coordinates
                x1_curr = (current.bbox[0] - current.bbox[2]/2)
                y1_curr = (current.bbox[1] - current.bbox[3]/2)
                x2_curr = (current.bbox[0] + current.bbox[2]/2)
                y2_curr = (current.bbox[1] + current.bbox[3]/2)
                
                x1_det = (det.bbox[0] - det.bbox[2]/2)
                y1_det = (det.bbox[1] - det.bbox[3]/2)
                x2_det = (det.bbox[0] + det.bbox[2]/2)
                y2_det = (det.bbox[1] + det.bbox[3]/2)
                
                # Calculate intersection
                x1 = max(x1_curr, x1_det)
                y1 = max(y1_curr, y1_det)
                x2 = min(x2_curr, x2_det)
                y2 = min(y2_curr, y2_det)
                
                if x2 < x1 or y2 < y1:
                    ious.append(0)
                    continue
                    
                intersection = (x2 - x1) * (y2 - y1)
                area_curr = (x2_curr - x1_curr) * (y2_curr - y1_curr)
                area_det = (x2_det - x1_det) * (y2_det - y1_det)
                union = area_curr + area_det - intersection
                
                ious.append(intersection / union if union > 0 else 0)
            
            # Remove detections with high IoU
            detections = [det for det, iou in zip(detections, ious) if iou < iou_threshold]
            
        return keep
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on a single image using patch-based processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of Detection objects
        """
        # Create patches
        patches, positions = self.create_patches(image)
        all_detections = []
        
        # Process each patch
        for patch, pos in zip(patches, positions):
            results = self.model(patch, conf=self.conf_threshold)[0]
            patch_detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Convert to normalized center coordinates
                patch_h, patch_w = patch.shape[:2]
                x_center = (x1 + x2) / (2 * patch_w)
                y_center = (y1 + y2) / (2 * patch_h)
                width = (x2 - x1) / patch_w
                height = (y2 - y1) / patch_h
                
                patch_detections.append(Detection(
                    class_id=cls,
                    confidence=conf,
                    bbox=(x_center, y_center, width, height),
                    tile_pos=pos
                ))
            
            # Convert to global coordinates
            global_detections = self.convert_to_global_coords(
                patch_detections, pos, patch.shape[:2], image.shape[:2]
            )
            all_detections.extend(global_detections)
        
        # Apply NMS
        return self.non_max_suppression(all_detections)
    
    def detect_batch(self, image_paths: List[str]) -> Dict[str, List[Detection]]:
        """
        Run detection on multiple images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Dictionary mapping image paths to their detections
        """
        results = {}
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue
                    
                detections = self.detect(image)
                results[img_path] = detections
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                
        return results

def visualize_detections(
    image: np.ndarray,
    detections: List[Detection],
    class_names: Dict[int, str],
    output_path: Optional[str] = None,
    show_labels: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Visualize detections on image.
    
    Args:
        image: Input image
        detections: List of Detection objects
        class_names: Dictionary mapping class IDs to names
        output_path: Path to save visualization (optional)
        show_labels: Whether to show class labels and confidence scores
        line_thickness: Thickness of bounding box lines
        
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
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)
        
        if show_labels:
            # Draw label
            label = f"{class_names[det.class_id]} {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness)
            cv2.rectangle(vis_image, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), line_thickness)
    
    # Save visualization if output path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image

def main():
    """Main function to run inference from command line."""
    parser = argparse.ArgumentParser(description="Run insect detection on images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--patch-size", type=int, default=864, help="Size of image patches")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = InsectDetector(args.model, args.conf, args.patch_size)
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    else:
        image_paths = [str(p) for p in input_path.glob("*.jpg")] + [str(p) for p in input_path.glob("*.png")]
    
    # Run detection
    results = detector.detect_batch(image_paths)
    
    # Visualize and save results
    for img_path, detections in results.items():
        image = cv2.imread(img_path)
        output_path = str(output_dir / Path(img_path).name)
        visualize_detections(image, detections, detector.class_names, output_path)
        
        # Save detection results as JSON
        json_path = str(output_dir / Path(img_path).stem) + "_detections.json"
        detections_dict = [
            {
                "class_id": det.class_id,
                "class_name": detector.class_names[det.class_id],
                "confidence": det.confidence,
                "bbox": det.bbox,
                "tile_pos": det.tile_pos
            }
            for det in detections
        ]
        with open(json_path, "w") as f:
            json.dump(detections_dict, f, indent=2)

if __name__ == "__main__":
    main() 