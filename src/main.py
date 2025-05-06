import argparse
from pathlib import Path
import json
from src.data.preprocessing import process_dataset
from src.data.dataset import split_dataset, process_dataset_for_tiling
from src.train import create_dataset_yaml, train_model, validate_model
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Insect Detection Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing original dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save all outputs')
    parser.add_argument('--model_size', type=str, default='m',
                      choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--tile_size', type=int, default=864,
                      help='Size of image tiles')
    parser.add_argument('--overlap', type=float, default=0.15,
                      help='Overlap ratio between tiles')
    parser.add_argument('--device', type=str, default='0',
                      help='Device to use for training/inference')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocess dataset
    print("Step 1: Preprocessing dataset...")
    processed_dir = output_dir / 'processed'
    process_dataset(args.data_dir, str(processed_dir))
    
    # Step 2: Split dataset
    print("Step 2: Splitting dataset...")
    split_dir = output_dir / 'split'
    split_dataset(str(processed_dir), str(split_dir))
    
    # Step 3: Create tiles
    print("Step 3: Creating tiles...")
    tiled_dir = output_dir / 'tiled'
    process_dataset_for_tiling(
        str(split_dir),
        str(tiled_dir),
        tile_size=args.tile_size,
        overlap=args.overlap
    )
    
    # Step 4: Create dataset YAML
    print("Step 4: Creating dataset configuration...")
    data_yaml = output_dir / 'dataset.yaml'
    create_dataset_yaml(str(tiled_dir), str(data_yaml))
    
    # Step 5: Train model
    print("Step 5: Training model...")
    model_dir = output_dir / 'model'
    train_model(
        str(data_yaml),
        str(model_dir),
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.tile_size,
        device=args.device
    )
    
    # Step 6: Validate model
    print("Step 6: Validating model...")
    val_dir = output_dir / 'validation'
    validate_model(
        str(model_dir / 'best.pt'),
        str(data_yaml),
        str(val_dir),
        batch_size=args.batch_size,
        img_size=args.tile_size,
        device=args.device
    )
    
    # Step 7: Evaluate model on test set
    print("Step 7: Evaluating model...")
    eval_dir = output_dir / 'evaluation'
    
    # Load class names
    with open(tiled_dir / 'class_mapping.json', 'r') as f:
        class_names = {int(k): v for k, v in json.load(f).items()}
    
    evaluate_model(
        str(model_dir / 'best.pt'),
        str(tiled_dir / 'test'),
        str(eval_dir),
        class_names,
        device=args.device
    )
    
    print("Pipeline completed successfully!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main() 