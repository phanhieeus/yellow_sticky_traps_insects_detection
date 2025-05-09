# Insect Detection on Yellow Sticky Traps

This project implements a full pipeline for detecting and classifying insects on yellow sticky traps using YOLOv8. The pipeline includes data preprocessing, model training, and evaluation.

## Setup

### Virtual Environment

It's recommended to use a virtual environment to manage dependencies. Here's how to set it up:

1. Create a virtual environment:
```bash
# Using venv (Python 3.3+)
python -m venv venv

# Or using conda
conda create -n insect_detection python=3.8
```

2. Activate the virtual environment:
```bash
# Using venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows

# Or using conda
conda activate insect_detection
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Deactivate when done:
```bash
deactivate  # For venv
# or
conda deactivate  # For conda
```

## Dataset

The project uses the [Yellow Sticky Traps dataset](https://www.kaggle.com/datasets/phnvnh/yellow-sticky-traps-dataset-vip) from Kaggle, which contains:
- 284 `.jpg` images
- 284 `.xml` annotations (Pascal VOC format)
- Three insect classes: Macrolophus (MR), Nesidiocoris (NC), and Whiteflies (WF)


### Downloading the Dataset

1. Create a Kaggle account if you don't have one
2. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Scroll to the "API" section
   - Click "Create New API Token"
   - This will download a `kaggle.json` file

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

The dataset will be downloaded and extracted to the `data/` directory.

### Dataset Structure and Annotation Conversion

After downloading, the dataset has the following structure:
```
data/yellow-sticky-traps-dataset-main/
├── images/           # Original .jpg images
├── annotations/      # Original .xml annotations (Pascal VOC format)
```

The original annotations are in Pascal VOC format (XML files). To use them with YOLO, we need to convert them to YOLO format. This is done using the `xml_to_yolo.py` script:

```bash
python src/data/xml_to_yolo.py
```

The script:
1. Reads all XML files from the `annotations` directory
2. Converts bounding box coordinates from Pascal VOC format (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)
3. Normalizes all coordinates to be between 0 and 1
4. Maps class names to IDs:
   - WF (Whiteflies): class_id 0
   - MR (Macrolophus): class_id 1
   - NC (Nesidiocoris): class_id 2
5. Saves the converted annotations as .txt files in the `labels` directory

Each .txt file contains one line per object in the format:
```
class_id x_center y_center width height
```

## Dataset Splitting

You can split the dataset into train and test sets using the provided script:

```bash
python src/data/split_dataset.py
```

This script will:
- Read all YOLO label files in `data/yellow-sticky-traps-dataset-main/labels/`
- Randomly split the dataset into train (90%) and test (10%) sets, ensuring the class distribution is preserved as much as possible
- Copy the corresponding images and label files into a new directory: `data/data_for_train_test/`
- Save the split information and class statistics to `data/data_for_train_test/split_info.json`

The resulting directory structure will be:
```
data/data_for_train_test/
├── train/
│   ├── images/   # Training images
│   └── labels/   # Training labels
├── test/
│   ├── images/   # Test images
│   └── labels/   # Test labels
└── split_info.json  # File listing which files are in train/test and class statistics
```

This makes it easy to train and evaluate models on consistent, balanced splits of your data.

### Dataset Analysis

The dataset can be analyzed using the provided script:

```bash
# Analyze the entire original dataset
python src/data/analyze_dataset.py --data_dir data --output_dir data/dataset_analysis_results

# Analyze only the train split
python src/data/analyze_dataset.py --data_dir data --labels_dir data/data_for_train_test/train/labels --output_dir data/train_analysis_results

# Analyze only the test split
python src/data/analyze_dataset.py --data_dir data --labels_dir data/data_for_train_test/test/labels --output_dir data/test_analysis_results
```

Each run will generate:
1. A detailed JSON statistics file (e.g., `train_stats.json`, `test_stats.json`, ...), including:
   - Total number of images and objects
   - Class distribution (WF, MR, NC)
   - Objects per image statistics (mean, median, min, max)
   - Object size statistics per class (normalized area)

2. Three visualization files in the output directory:
   - `train_class_distribution.png`, `test_class_distribution.png`, ...: Bar plot showing class distribution
   - `train_objects_per_image.png`, `test_objects_per_image.png`, ...: Histogram of objects per image
   - `train_object_sizes.png`, `test_object_sizes.png`, ...: Boxplot of object sizes by class


### Image Tiling

Because the original images are very large (5184x3456) and the insects are very small, training directly on the full images is not effective. To address this, you can split each large training image into smaller tiles (patches) of size 864x864 pixels with 20% overlap. This ensures that small insects near the edges are not lost and improves detection performance for small objects.

The tiling process:
- Splits each training image into multiple 864x864 tiles, with each tile overlapping its neighbors by 20%.
- For each tile, only the bounding boxes (insects) that fall within the tile are kept, and their coordinates are updated relative to the tile.
- Each tile is saved as a new image, named as `originalname_1.jpg`, `originalname_2.jpg`, etc., with a corresponding YOLO label file.
- The results are saved in:
  - `data/tiled_data/train/images/`
  - `data/tiled_data/train/labels/`

To run the tiling script:
```bash
python src/data/tile_images.py
```

### Visualizing Tiled Images and Labels

To visually check the correctness of the tiling and label conversion, you can use the provided visualization script. This script draws bounding boxes on a random sample of tiled images and saves the results for inspection.

Example usage:
```bash
python src/utils/visualize_tile.py \
  --tile_dir data/tiled_data/train/images \
  --label_dir data/tiled_data/train/labels \
  --output_dir visualized_tiles \
  --num 10
```
- This will randomly select 10 tiles, draw their bounding boxes, and save the visualized images to a new folder named `visualized_tiles/` in the project root.
- If the folder already exists, the script will automatically create a new one with a suffix (e.g., `visualized_tiles_vis1`).
- Box colors: WF (green), MR (red), NC (blue), with class names shown on each box.

### Splitting Tiled Dataset for Training YOLOv8

After tiling, you should split the resulting tiles into train and test sets for YOLOv8 training. To ensure a balanced class distribution, use the provided script to split the tiles so that each class is represented proportionally in both sets (train:test ratio 9:1).

The script will:
- Analyze all tiled label files in `data/tiled_data/train/labels/` to count the number of objects per class in each tile.
- Split the tiles into train and test sets, aiming to balance the number of objects of each class between the two sets as much as possible.
- Copy the corresponding images and label files into the following structure:

```
data/tiled_data/
├── train_split/
│   ├── images/   # Training tiles
│   └── labels/   # Training labels
├── test/
│   ├── images/   # Test tiles
│   └── labels/   # Test labels
```

To run the script:
```bash
python src/data/split_tiled_dataset.py
```

After running, the script will print statistics for each set (number of tiles and number of objects per class) and ensure the splits are ready for YOLOv8 training.

### Training YOLOv8 on Tiled Dataset

After splitting the tiled dataset, you can train a YOLOv8 model using the provided training script. The script will automatically create a `dataset.yaml` file that points to the tiled train and validation sets, and only includes `train` and `val` (no `test`), as you may want to test on the original full-size images separately.

The training script will:
- Use the tiles in `data/tiled_data/train_split/images` and `labels` for training.
- Use the tiles in `data/tiled_data/test/images` and `labels` as the validation set (`val`).
- Automatically generate a `dataset.yaml` file in `data/tiled_data/` with the correct structure for YOLOv8.
- Train the model and save the best weights to your specified output directory.
- Run validation on the validation set after training.
- Fully utilize the GPU if you set `--device 0` (or another GPU index).

Example usage:
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
- The script will print progress and save results in the `runs/yolov8/` directory.
- You can later evaluate the trained model on the original (non-tiled) test set using a separate validation/inference script.

After training completes, you will get:
- `best.pt`: The best model weights saved in your output directory (e.g., `runs/yolov8/best.pt`). This file can be used for inference or further evaluation.
- Training logs and metrics: YOLOv8 will save training progress, loss curves, and metrics (such as mAP, precision, recall) in the output directory for each run.
- Validation results: After training, the script will automatically run validation on the validation set and save the results (including per-class metrics and confusion matrix) in the output directory.
- All results are organized under your specified `--output_dir` (e.g., `runs/yolov8/`).

You can use the `best.pt` model for inference on new images or for further evaluation on the original (non-tiled) test set.




## Features

- Data preprocessing and analysis:
  - XML to YOLO format annotation conversion
  - Detailed dataset statistics and visualizations
  - Class distribution analysis
  - Object size and density analysis

- Dataset preparation:
  - Dataset splitting with class balance preservation
  - Image tiling with overlap for small object detection
  - Automatic handling of multi-scale objects

- Model training and inference:
  - YOLOv8 model training with GPU support
  - Full-size image inference with tiling
  - Non-maximum suppression (NMS) for overlapping detections

- Evaluation and visualization:
  - Per-class precision, recall, and F1-score
  - Confusion matrix generation
  - Detection visualization tools
  - Performance metrics tracking



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

