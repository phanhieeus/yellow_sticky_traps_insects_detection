import os
import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 864
OVERLAP = 0.2  # 20%

# Input/output paths
TRAIN_IMG_DIR = 'data/data_for_train_test/train/images'
TRAIN_LABEL_DIR = 'data/data_for_train_test/train/labels'
OUT_IMG_DIR = 'data/tiled_data/images'
OUT_LABEL_DIR = 'data/tiled_data/labels'

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

def load_labels(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            # Convert to (xmin, ymin, xmax, ymax)
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            boxes.append([class_id, xmin, ymin, xmax, ymax])
    return boxes

def save_labels(label_path, boxes, tile_w, tile_h):
    with open(label_path, 'w') as f:
        for box in boxes:
            class_id, xmin, ymin, xmax, ymax = box
            # Convert back to YOLO format (normalized)
            x_center = (xmin + xmax) / 2 / tile_w
            y_center = (ymin + ymax) / 2 / tile_h
            width = (xmax - xmin) / tile_w
            height = (ymax - ymin) / tile_h
            # Only keep boxes with positive area
            if width > 0 and height > 0:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def tile_image(img, boxes, img_name, out_img_dir, out_label_dir, tile_size=IMG_SIZE, overlap=OVERLAP):
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tile_id = 1
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            # If tile is smaller than tile_size, pad it
            tile = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
            tile_img = img[y0:y1, x0:x1]
            tile[:y1-y0, :x1-x0] = tile_img
            # Adjust boxes for this tile
            tile_boxes = []
            for box in boxes:
                class_id, xmin, ymin, xmax, ymax = box
                # Check if the object is fully inside the tile (not cut)
                if xmin >= x0 and ymin >= y0 and xmax <= x1 and ymax <= y1:
                    # Box coordinates relative to tile
                    new_xmin = xmin - x0
                    new_ymin = ymin - y0
                    new_xmax = xmax - x0
                    new_ymax = ymax - y0
                    tile_boxes.append([class_id, new_xmin, new_ymin, new_xmax, new_ymax])
            # Save tile and label only if at least one fully-contained box
            if len(tile_boxes) > 0:
                tile_img_name = f"{Path(img_name).stem}_{tile_id}.jpg"
                tile_label_name = f"{Path(img_name).stem}_{tile_id}.txt"
                cv2.imwrite(os.path.join(out_img_dir, tile_img_name), tile)
                save_labels(os.path.join(out_label_dir, tile_label_name), tile_boxes, tile_size, tile_size)
                tile_id += 1

def main():
    img_files = sorted(os.listdir(TRAIN_IMG_DIR))
    for img_file in img_files:
        img_path = os.path.join(TRAIN_IMG_DIR, img_file)
        label_path = os.path.join(TRAIN_LABEL_DIR, Path(img_file).with_suffix('.txt'))
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        boxes = load_labels(label_path, w, h)
        tile_image(img, boxes, img_file, OUT_IMG_DIR, OUT_LABEL_DIR)
        print(f"Tiled {img_file}")

if __name__ == '__main__':
    main() 