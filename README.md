# Insect Detection on Yellow Sticky Traps

This project implements a full pipeline for detecting and classifying insects on yellow sticky traps using YOLOv8. The pipeline includes data preprocessing, model training, and evaluation.

## Dataset

The project uses the [Yellow Sticky Traps dataset](https://www.kaggle.com/datasets/friso1987/yellow-sticky-traps) from Kaggle, which contains:
- 284 `.jpg` images
- 284 `.xml` annotations (Pascal VOC format)
- Three insect classes: Macrolophus (MR), Nesidiocoris (NC), and Whiteflies (WF)

## Features

- Image rotation and annotation conversion
- Dataset splitting with class balance preservation
- Image tiling with overlap for small object detection
- YOLOv8 model training with GPU support
- Full-size image inference with tiling and NMS
- Comprehensive evaluation metrics and visualizations

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Image rotation and annotation conversion
│   │   └── dataset.py       # Dataset splitting and tiling
│   ├── train.py             # Model training
│   ├── inference.py         # Inference on full-size images
│   ├── evaluation.py        # Evaluation metrics and visualizations
│   └── main.py             # Main pipeline script
├── requirements.txt
└── README.md
```

## Usage

1. Download the dataset from Kaggle and extract it to a directory.

2. Run the full pipeline:

```bash
python src/main.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/output \
    --model_size m \
    --epochs 100 \
    --batch_size 16 \
    --tile_size 864 \
    --overlap 0.15 \
    --device 0
```

Arguments:
- `--data_dir`: Directory containing the original dataset
- `--output_dir`: Directory to save all outputs
- `--model_size`: YOLOv8 model size (n, s, m, l, x)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--tile_size`: Size of image tiles
- `--overlap`: Overlap ratio between tiles
- `--device`: Device to use for training/inference (0 for GPU, cpu for CPU)

## Pipeline Steps

1. **Preprocessing**:
   - Rotate images 90 degrees clockwise
   - Convert VOC annotations to YOLO format

2. **Dataset Splitting**:
   - Split into train/val/test sets
   - Maintain class distribution

3. **Tiling**:
   - Create overlapping tiles (864x864 pixels)
   - Adjust bounding boxes for tiles
   - Filter out small boxes

4. **Training**:
   - Train YOLOv8 model on tiles
   - Save best model weights

5. **Validation**:
   - Validate model on validation set
   - Generate validation metrics

6. **Evaluation**:
   - Evaluate on test set
   - Generate confusion matrix
   - Calculate per-class metrics

## Output

The pipeline generates the following outputs in the specified output directory:

```
output_dir/
├── processed/          # Rotated images and converted annotations
├── split/             # Train/val/test splits
├── tiled/             # Tiled images and annotations
├── dataset.yaml       # Dataset configuration
├── model/             # Trained model and weights
├── validation/        # Validation results
└── evaluation/        # Evaluation metrics and visualizations
```

## Evaluation Metrics

The evaluation includes:
- Per-class precision, recall, and F1-score
- Confusion matrix
- Visualizations of detections
- JSON files with detailed metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details. 