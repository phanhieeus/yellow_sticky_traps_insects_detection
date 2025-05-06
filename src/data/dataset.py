import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from sklearn.model_selection import train_test_split
import json

def split_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Split dataset into train, validation and test sets while maintaining class distribution.
    
    Args:
        data_dir: Directory containing processed dataset
        output_dir: Directory to save split datasets
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create split directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list((data_dir / 'images').glob('*.jpg'))
    
    # First split into train and temp
    train_files, temp_files = train_test_split(
        image_files,
        train_size=train_ratio,
        random_state=42,
        shuffle=True
    )
    
    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        random_state=42,
        shuffle=True
    )
    
    # Copy files to respective directories
    for split, files in zip(splits, [train_files, val_files, test_files]):
        for img_path in files:
            # Copy image
            shutil.copy2(
                img_path,
                output_dir / split / 'images' / img_path.name
            )
            
            # Copy corresponding label
            label_path = data_dir / 'labels' / img_path.with_suffix('.txt').name
            if label_path.exists():
                shutil.copy2(
                    label_path,
                    output_dir / split / 'labels' / label_path.name
                )
    
    # Copy class mapping
    shutil.copy2(
        data_dir / 'class_mapping.json',
        output_dir / 'class_mapping.json'
    )

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

def adjust_bbox_for_tile(bbox: Tuple[float, float, float, float], 
                        tile_pos: Tuple[int, int],
                        tile_size: int,
                        img_width: int,
                        img_height: int) -> Tuple[float, float, float, float]:
    """
    Adjust bbox coordinates for a tile.
    
    Args:
        bbox: YOLO format bbox (x_center, y_center, width, height)
        tile_pos: (x, y) position of tile
        tile_size: Size of tile
        img_width, img_height: Original image dimensions
        
    Returns:
        Adjusted bbox in YOLO format
    """
    x_center, y_center, width, height = bbox
    
    # Convert to absolute coordinates
    x_abs = x_center * img_width
    y_abs = y_center * img_height
    
    # Adjust for tile position
    x_adj = x_abs - tile_pos[0]
    y_adj = y_abs - tile_pos[1]
    
    # Convert back to relative coordinates
    x_center_adj = x_adj / tile_size
    y_center_adj = y_adj / tile_size
    width_adj = width * img_width / tile_size
    height_adj = height * img_height / tile_size
    
    return x_center_adj, y_center_adj, width_adj, height_adj

def process_image_for_tiling(img_path: str,
                           label_path: str,
                           output_dir: str,
                           tile_size: int = 864,
                           overlap: float = 0.15,
                           min_box_size: int = 10):
    """
    Process a single image and its annotations for tiling.
    
    Args:
        img_path: Path to image
        label_path: Path to YOLO format label file
        output_dir: Directory to save tiles
        tile_size: Size of square tiles
        overlap: Overlap ratio between tiles
        min_box_size: Minimum box size in pixels to keep
    """
    # Read image
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    
    # Read annotations
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, *bbox = map(float, line.strip().split())
            annotations.append((int(class_id), bbox))
    
    # Create tiles
    tiles = create_tiles(image, tile_size, overlap)
    
    # Process each tile
    for i, (tile, (x, y)) in enumerate(tiles):
        tile_annotations = []
        
        # Process each annotation
        for class_id, bbox in annotations:
            # Adjust bbox for tile
            bbox_adj = adjust_bbox_for_tile(bbox, (x, y), tile_size, width, height)
            
            # Check if box is valid
            x_center, y_center, w, h = bbox_adj
            if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                w * tile_size >= min_box_size and h * tile_size >= min_box_size):
                tile_annotations.append((class_id, bbox_adj))
        
        # Save tile and annotations if it contains any objects
        if tile_annotations:
            # Save tile
            tile_path = Path(output_dir) / 'images' / f"{Path(img_path).stem}_tile_{i}.jpg"
            cv2.imwrite(str(tile_path), tile)
            
            # Save annotations
            label_path = Path(output_dir) / 'labels' / f"{Path(img_path).stem}_tile_{i}.txt"
            with open(label_path, 'w') as f:
                for class_id, bbox in tile_annotations:
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

def process_dataset_for_tiling(data_dir: str,
                             output_dir: str,
                             tile_size: int = 864,
                             overlap: float = 0.15,
                             min_box_size: int = 10):
    """
    Process entire dataset for tiling.
    
    Args:
        data_dir: Directory containing split dataset
        output_dir: Directory to save tiled dataset
        tile_size: Size of square tiles
        overlap: Overlap ratio between tiles
        min_box_size: Minimum box size in pixels to keep
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        for img_path in (split_dir / 'images').glob('*.jpg'):
            label_path = split_dir / 'labels' / img_path.with_suffix('.txt').name
            if label_path.exists():
                process_image_for_tiling(
                    str(img_path),
                    str(label_path),
                    str(output_dir / split),
                    tile_size,
                    overlap,
                    min_box_size
                )
    
    # Copy class mapping
    shutil.copy2(
        data_dir / 'class_mapping.json',
        output_dir / 'class_mapping.json'
    ) 