import os
import cv2
import argparse
from pathlib import Path

def draw_boxes(image_path, label_path, output_path, class_names=None):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    color_map = [(0,255,0), (255,0,0), (0,0,255)]  # WF: green, MR: red, NC: blue
    if class_names is None:
        class_names = ['WF', 'MR', 'NC']
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h
                xmin = int(x_center - width/2)
                ymin = int(y_center - height/2)
                xmax = int(x_center + width/2)
                ymax = int(y_center + height/2)
                color = color_map[class_id % len(color_map)]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, class_names[class_id], (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(output_path), img)

def get_unique_output_dir(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return base
    i = 1
    while True:
        new_dir = base.parent / f"{base.name}_vis{i}"
        if not new_dir.exists():
            return new_dir
        i += 1

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO tiles with bounding boxes')
    parser.add_argument('--tile_dir', type=str, required=True, help='Directory containing tile images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing tile labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualized images (e.g. visualized_tiles)')
    parser.add_argument('--num', type=int, default=10, help='Number of tiles to visualize')
    args = parser.parse_args()

    # Make output dir unique if exists
    output_dir = get_unique_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tile_files = sorted([f for f in os.listdir(args.tile_dir) if f.endswith('.jpg')])
    if args.num > 0:
        import random
        random.seed(42)
        tile_files = random.sample(tile_files, min(args.num, len(tile_files)))
    for tile_file in tile_files:
        img_path = Path(args.tile_dir) / tile_file
        label_path = Path(args.label_dir) / (Path(tile_file).stem + '.txt')
        out_path = output_dir / tile_file
        draw_boxes(img_path, label_path, out_path)
        print(f"Visualized {tile_file} -> {out_path}")

if __name__ == '__main__':
    print("Example usage:")
    print("python src/utils/visualize_tile.py --tile_dir data/tiled_data/train/images --label_dir data/tiled_data/train/labels --output_dir visualized_tiles --num 10")
    main() 