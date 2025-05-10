import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import os
import json
from tqdm import tqdm

PATCH_SIZE = 864

class_names = {
    0: "WF",  # Whiteflies
    1: "MR",  # Macrolophus
    2: "NC"   # Nesidiocoris
}

colors = {
    0: (0, 255, 0),    # Green for WF (Whiteflies)
    1: (255, 0, 0),    # Red for MR (Macrolophus)
    2: (0, 0, 255)     # Blue for NC (Nesidiocoris)
}

def create_patches(image, patch_size=PATCH_SIZE):
    h, w = image.shape[:2]
    patches = []
    positions = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

def apply_nms(boxes, scores, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    boxes: list of [x1, y1, x2, y2]
    scores: list of confidence scores
    iou_threshold: IoU threshold for NMS
    """
    if len(boxes) == 0:
        return []

    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by confidence score
    indices = np.argsort(scores)[::-1]

    keep = []
    while indices.size > 0:
        # Pick the box with highest confidence
        i = indices[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h

        # Calculate IoU
        iou = intersection / (areas[i] + areas[indices[1:]] - intersection)

        # Remove boxes with IoU > threshold
        indices = indices[1:][iou <= iou_threshold]

    return keep

def draw_detections(image, detections, positions, patch_size=PATCH_SIZE):
    h_img, w_img = image.shape[:2]
    
    # Collect all boxes and scores
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for det, (x0, y0) in zip(detections, positions):
        for box in det.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Convert to global coordinates
            x1g = int(x1 + x0)
            y1g = int(y1 + y0)
            x2g = int(x2 + x0)
            y2g = int(y2 + y0)
            
            all_boxes.append([x1g, y1g, x2g, y2g])
            all_scores.append(conf)
            all_classes.append(cls)
    
    # Apply NMS
    if all_boxes:
        keep_indices = apply_nms(all_boxes, all_scores)
        
        # Draw only the boxes that passed NMS
        for idx in keep_indices:
            x1g, y1g, x2g, y2g = all_boxes[idx]
            conf = all_scores[idx]
            cls = all_classes[idx]
            
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(image, (x1g, y1g), (x2g, y2g), color, 2)
            label = f"{class_names[cls]} {conf:.2f}"
            cv2.putText(image, label, (x1g, y1g-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def save_detections_to_json(detections, positions, image_path, output_dir):
    """
    Save detection results to a JSON file
    """
    results = {
        "image_path": image_path,
        "detections": []
    }
    
    # Collect all boxes and scores
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for det, (x0, y0) in zip(detections, positions):
        for box in det.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Convert to global coordinates
            x1g = int(x1 + x0)
            y1g = int(y1 + y0)
            x2g = int(x2 + x0)
            y2g = int(y2 + y0)
            
            all_boxes.append([x1g, y1g, x2g, y2g])
            all_scores.append(conf)
            all_classes.append(cls)
    
    # Apply NMS
    if all_boxes:
        keep_indices = apply_nms(all_boxes, all_scores)
        
        # Save only the boxes that passed NMS
        for idx in keep_indices:
            x1g, y1g, x2g, y2g = all_boxes[idx]
            conf = all_scores[idx]
            cls = all_classes[idx]
            
            detection = {
                "bbox": [x1g, y1g, x2g, y2g],
                "confidence": float(conf),
                "class": int(cls),
                "class_name": class_names[cls]
            }
            results["detections"].append(detection)
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{Path(image_path).stem}_det.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return json_path

def process_image(image_path, model, output_dir, conf_threshold=0.35):
    """
    Xử lý một ảnh và lưu kết quả
    """
    # Đọc ảnh
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    # Cắt patch
    patches, positions = create_patches(image, PATCH_SIZE)

    # Detect từng patch với confidence threshold
    detections = [model(patch, conf=conf_threshold)[0] for patch in patches]

    # Vẽ kết quả lên ảnh gốc
    vis_image = image.copy()
    vis_image = draw_detections(vis_image, detections, positions, PATCH_SIZE)

    # Lưu kết quả ảnh
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{image_path.stem}_det.jpg")
    cv2.imwrite(out_path, vis_image)

    # Lưu kết quả JSON
    json_path = save_detections_to_json(detections, positions, str(image_path), output_dir)
    
    return json_path

def main():
    parser = argparse.ArgumentParser(description="Inference on test images with patching.")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Xử lý input là file hoặc thư mục
    input_path = Path(args.input)
    if input_path.is_file():
        # Xử lý một ảnh
        json_path = process_image(input_path, model, args.output, args.conf)
        if json_path:
            print(f"Result saved to {json_path}")
    else:
        # Xử lý toàn bộ thư mục
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + list(input_path.glob('*.png'))
        if not image_files:
            print(f"No image files found in {args.input}")
            return

        print(f"Processing {len(image_files)} images...")
        for image_path in tqdm(image_files):
            json_path = process_image(image_path, model, args.output, args.conf)
            if json_path:
                print(f"Processed {image_path.name} -> {json_path}")

if __name__ == "__main__":
    main() 