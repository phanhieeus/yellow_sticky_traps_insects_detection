# Insect Detection on Yellow Sticky Traps

This project implements a full pipeline for detecting and classifying insects on yellow sticky traps using YOLOv8. The pipeline includes data preprocessing, model training, and evaluation.

## Overview

The project aims to detect and classify three types of insects on yellow sticky traps:
- Whiteflies (WF)
- Macrolophus (MR)
- Nesidiocoris (NC)

## Model

The project uses YOLOv8m (medium) model trained on:
- Hardware: NVIDIA RTX 4090
- Training time: 1 hour
- Epochs: 50
- Model size: 50MB
- Location: `final_model/best.pt`


### Sample Results

The repository includes sample detection results in the `inference_results/` directory:
- 29 test images with detections
- Each image has two associated files:
  - `*_det.jpg`: Visualization with bounding boxes
  - `*_det.json`: Detailed detection results including:
    - Class ID and name
    - Confidence score
    - Bounding box coordinates
    - Tile position
- Evaluation metrics saved in `inference_results/eval_results.json`

## Evaluation Results (Test set)

**Overall Metrics:**
- Total Ground Truth Objects: 1148
- Total Detected Objects: 1146
- True Positives: 1015
- False Positives: 131
- False Negatives: 133
- Precision: 0.8857
- Recall: 0.8841
- F1 Score: 0.8849
- mAP50: 0.7860

**Per-Class Metrics:**
- **WF**: Precision 0.8978, Recall 0.8479, F1 0.8721, mAP50 0.7736
- **MR**: Precision 0.8850, Recall 0.8850, F1 0.8850, mAP50 0.8020
- **NC**: Precision 0.7391, Recall 1.0000, F1 0.8500, mAP50 0.7823

## Features

- Dataset preprocessing
- Image tiling for small object detection
- Dataset analysis and visualization
- YOLOv8 model training and evaluation
- Inference on full-size images
- Visualization tools
- Comprehensive evaluation metrics

## Project Structure

```
yellow_sticky_traps_insects_detection/
├── src/                    # Source code
│   ├── data/              # Data processing scripts
│   │   ├── xml_to_yolo.py        # Convert Pascal VOC to YOLO format
│   │   ├── split_dataset.py      # Split dataset into train/test
│   │   ├── tile_images.py        # Split large images into tiles
│   │   ├── split_tiled_dataset.py # Split tiled dataset
│   │   └── analyze_dataset.py    # Dataset analysis tools
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   ├── evaluation.py      # Evaluation script
│   └── utils/             # Utility functions
│       └── visualize_tile.py     # Visualization tools
├── data/                  # Dataset and processed data
│   ├── yellow-sticky-traps-dataset-main/  # Original dataset
│   ├── data_for_train_test/              # Split dataset
│   └── tiled_data/                       # Tiled dataset
│       ├── train/                        # Training tiles
│       └── test/                         # Test tiles
├── final_model/          # Trained model
│   └── best.pt           # Best model weights (50MB)
├── inference_results/    # Detection results
│   ├── *_det.jpg        # Visualization images
│   ├── *_det.json       # Detection details
│   └── eval_results.json # Evaluation metrics
├── requirements.txt       # Python dependencies
├── README.md             # Project overview
└── GETTING_STARTED.md    # Detailed instructions
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/phanhieeus/yellow_sticky_traps_insects_detection.git
cd yellow_sticky_traps_insects_detection
```

For detailed instructions on dataset preparation, training, and evaluation, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:
```
@misc{yellow_sticky_traps_insects_detection,
  author = {Phan Hieu},
  title = {Yellow Sticky Traps Insects Detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/phanhieeus/yellow_sticky_traps_insects_detection}}
}
```




