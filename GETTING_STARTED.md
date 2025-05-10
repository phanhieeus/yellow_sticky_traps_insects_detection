# Getting Started with Yellow Sticky Traps Insects Detection

This guide will walk you through the process of setting up and using the insect detection project from scratch.

## 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/phanhieeus/yellow_sticky_traps_insects_detection.git
cd yellow_sticky_traps_insects_detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Dataset

You have two options to download the dataset:

### Option 1: Download from Kaggle

1. Create a Kaggle account if you don't have one
2. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. Set up Kaggle credentials:
```bash
mkdir -p ~/.kaggle
cp path/to/downloaded/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

4. Run the download script:
```bash
./download_dataset.sh
```

### Option 2: Download from Google Drive

Simply run the download script:
```bash
./download_dataset_gdrive.sh
```

Both options will:
- Download the dataset
- Extract it to the correct directory structure
- Verify the dataset structure is correct
- The dataset will be available at: `data/yellow-sticky-traps-dataset-main/`

## 3. Data Preparation

1. Convert annotations from VOC to YOLO format:
```bash
python src/data/xml_to_yolo.py
```

2. Split dataset into train/test sets:
```bash
python src/data/split_dataset.py
```

3. Create tiles for training:
```bash
python src/data/tile_images.py
```

4. Split tiled dataset:
```bash
python src/data/split_tiled_dataset.py
```

## 4. Training

Train the YOLOv8 model on the tiled dataset:

```bash
python src/train.py \
    --data_dir data/tiled_data \
    --output_dir runs/yolov8 \
    --model_size m \
    --epochs 100 \
    --batch_size 16 \
    --img_size 864 \
    --device 0
```

### Training Parameters

You can customize the training process by adjusting these parameters:

- `--data_dir`: Directory containing the dataset (default: data/tiled_data)
- `--output_dir`: Directory to save model weights and logs (default: runs/yolov8)
- `--model_size`: YOLOv8 model size (n, s, m, l, x) (default: m)
  - n: nano (fastest, lowest accuracy)
  - s: small
  - m: medium (balanced)
  - l: large
  - x: xlarge (slowest, highest accuracy)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
  - Increase if you have more GPU memory
  - Decrease if you get CUDA out of memory errors
- `--img_size`: Input image size (default: 864)
  - Must match the tile size used in data preparation
- `--device`: GPU device index (default: 0)
  - Use -1 for CPU (not recommended)
  - Use 0, 1, etc. for specific GPUs

The training script will:
- Use tiles in `data/tiled_data/train/` for training
- Use tiles in `data/tiled_data/test/` for validation
- Save the best model as `runs/yolov8/best.pt`
- Generate training logs and metrics in `runs/yolov8/`

## 5. Inference

Run inference on images using the trained model. The script supports both single image and directory processing:

```bash
# For a single image
python src/inference.py \
    --model runs/yolov8/best.pt \
    --input path/to/image.jpg \
    --conf 0.35 \
    --output inference_results

# For a directory of images
python src/inference.py \
    --model runs/yolov8/best.pt \
    --input path/to/image/directory \
    --conf 0.35 \
    --output inference_results
```

### Inference Parameters

Customize the inference process with these parameters:

- `--model`: Path to the trained model file (default: runs/yolov8/best.pt)
- `--input`: Path to input image or directory containing images
- `--output`: Directory to save results (default: inference_results)
- `--conf`: Confidence threshold (default: 0.35)
  - Increase for fewer but more confident detections
  - Decrease for more detections but potentially more false positives
- `--iou`: IoU threshold for NMS (default: 0.45)
  - Increase to be more strict about overlapping detections
  - Decrease to allow more overlapping detections
- `--device`: Device to run inference on (default: 0)
  - Use -1 for CPU (not recommended)
  - Use 0, 1, etc. for specific GPUs

The inference script will:
1. Process each image by splitting it into patches of size 864x864
2. Run detection on each patch with confidence threshold of 0.35
3. Convert patch coordinates to global image coordinates
4. Apply Non-Maximum Suppression (NMS) with IoU threshold of 0.45 to merge overlapping detections
5. Save results in `inference_results/`:
   - `*_det.jpg`: Visualization with bounding boxes
   - `*_det.json`: Detailed detection results including:
     - Class ID and name
     - Confidence score
     - Bounding box coordinates (absolute pixel coordinates)
     - Tile position

## 6. Evaluation

Evaluate the model's performance on the test set:

```bash
python src/evaluation.py \
    --gt data/data_for_train_test/test/labels \
    --pred inference_results \
    --images data/data_for_train_test/test/images \
    --output inference_results/eval_results.json
```

### Evaluation Parameters

Customize the evaluation process with these parameters:

- `--gt`: Path to directory containing ground truth labels (YOLO format .txt)
- `--pred`: Path to directory containing detection results (JSON files)
- `--images`: Path to directory containing test images
- `--output`: JSON file to save evaluation results (default: eval_results.json)
- `--iou`: IoU threshold for determining true positives (default: 0.5)
  - Increase for stricter matching between predictions and ground truth
  - Decrease for more lenient matching
- `--conf`: Minimum confidence threshold for predictions (default: 0.001)
  - Only affects evaluation, not the actual predictions
  - Use to analyze model performance at different confidence levels

The evaluation script will:
- Compare detections with ground truth labels
- Calculate metrics per class and overall:
  - True Positives (TP)
  - False Positives (FP)
  - False Negatives (FN)
  - Precision
  - Recall
  - F1 Score
- Save detailed results in `inference_results/eval_results.json`

## Notes

- All data files (images, labels, results) are ignored by git
- Model weights and training logs are saved in `runs/`
- Inference results are automatically saved in `inference_results/`
- Evaluation results are saved in `inference_results/eval_results.json`
- Use `dataset.yaml` for training configuration

## Troubleshooting

1. If you get CUDA out of memory errors:
   - Reduce batch size (e.g., --batch_size 8)
   - Reduce patch size (e.g., --img_size 640)
   - Use a smaller model (e.g., --model_size s)
   - Close other GPU applications

2. If inference is slow:
   - Reduce patch size (e.g., --img_size 640)
   - Use a smaller model (e.g., --model_size s)
   - Process fewer images at once
   - Use a more powerful GPU

3. If detection quality is poor:
   - Adjust confidence threshold (default: 0.35)
     - Increase for fewer but more confident detections
     - Decrease for more detections but potentially more false positives
   - Adjust IoU threshold for NMS (default: 0.45)
     - Increase to be more strict about overlapping detections
     - Decrease to allow more overlapping detections
   - Increase training epochs (e.g., --epochs 200)
   - Use a larger model (e.g., --model_size l)
   - Check dataset quality and annotations 