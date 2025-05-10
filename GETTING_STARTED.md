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

The training script will:
- Use tiles in `data/tiled_data/train_split/` for training
- Use tiles in `data/tiled_data/test/` for validation
- Save the best model as `runs/yolov8/best.pt`
- Generate training logs and metrics in `runs/yolov8/`

## 5. Inference

Run inference on the test set using the trained model:

```bash
python src/inference.py \
    --model runs/yolov8/best.pt \
    --input data/data_for_train_test/test/images \
    --output inference_results \
    --tile_size 864 \
    --overlap 0.2 \
    --conf 0.25 \
    --iou 0.5 \
    --device 0
```

This will:
- Process each test image by splitting it into tiles
- Run detection on each tile
- Merge results and apply NMS
- Save results in `inference_results/`:
  - `*_det.jpg`: Images with bounding boxes
  - `*_det.json`: Detection results in JSON format

## 6. Evaluation

Evaluate the model's performance on the test set:

```bash
python src/evaluation.py \
    --gt data/data_for_train_test/test/labels \
    --pred inference_results \
    --output eval_results.json \
    --iou 0.5 \
    --conf 0.001
```

The evaluation script will:
- Compare detections with ground truth labels
- Calculate mAP@0.5, AP per class, precision, recall, and F1-score
- Save results in `eval_results.json`

## 7. Visualizing Results

To visualize the tiled images and their labels:

```bash
python src/utils/visualize_tile.py \
    --tile_dir data/tiled_data/train/images \
    --label_dir data/tiled_data/train/labels \
    --output_dir visualized_tiles \
    --num 10
```

## Project Structure

After setup, your project should have this structure:
```
yellow_sticky_traps_insects_detection/
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   └── utils/
│       └── visualize_tile.py
├── data/
│   ├── data_for_train_test/
│   │   ├── train/
│   │   └── test/
│   └── tiled_data/
│       ├── train_split/
│       └── test/
├── requirements.txt
├── README.md
└── GETTING_STARTED.md
```

## Notes

- All data files (images, labels, results) are ignored by git
- Model weights and training logs are saved in `runs/`
- Inference results are saved in `inference_results/`
- Evaluation results are saved in `eval_results.json`
- Use `dataset.yaml` for training configuration

## Troubleshooting

1. If you get CUDA out of memory errors:
   - Reduce batch size
   - Reduce image size
   - Use a smaller model (e.g., 's' instead of 'm')

2. If inference is slow:
   - Reduce tile size
   - Reduce overlap
   - Use a smaller model

3. If detection quality is poor:
   - Increase training epochs
   - Adjust confidence threshold
   - Adjust IoU threshold
   - Use a larger model 